#include "penta/osc/OSCServer.h"
#include "penta/osc/OSCClient.h"
#include "penta/osc/OSCHub.h"
#include "penta/osc/RTMessageQueue.h"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <variant>
#include <atomic>
#include <iostream>

using namespace penta::osc;

// ========== RTMessageQueue Tests ==========

class RTMessageQueueTest : public ::testing::Test
{
protected:
    RTMessageQueue queue{1024};
};

TEST_F(RTMessageQueueTest, PushAndPop)
{
    OSCMessage msg;
    msg.setAddress("/test");
    msg.addFloat(42.0f);

    EXPECT_TRUE(queue.push(msg));

    OSCMessage retrieved;
    EXPECT_TRUE(queue.pop(retrieved));
    EXPECT_EQ(retrieved.getAddress(), "/test");
    EXPECT_EQ(retrieved.getArgumentCount(), 1u);
    EXPECT_FLOAT_EQ(std::get<float>(retrieved.getArgument(0)), 42.0f);
}

TEST_F(RTMessageQueueTest, FIFOOrder)
{
    OSCMessage msg1, msg2, msg3;
    msg1.setAddress("/first");
    msg2.setAddress("/second");
    msg3.setAddress("/third");

    queue.push(msg1);
    queue.push(msg2);
    queue.push(msg3);

    OSCMessage retrieved;
    queue.pop(retrieved);
    EXPECT_EQ(retrieved.getAddress(), "/first");

    queue.pop(retrieved);
    EXPECT_EQ(retrieved.getAddress(), "/second");

    queue.pop(retrieved);
    EXPECT_EQ(retrieved.getAddress(), "/third");
}

TEST_F(RTMessageQueueTest, EmptyQueueReturnsFalse)
{
    OSCMessage msg;
    EXPECT_FALSE(queue.pop(msg));
}

TEST_F(RTMessageQueueTest, ClearWorks)
{
    OSCMessage msg;
    msg.setAddress("/test");

    queue.push(msg);

    // No explicit clear() in current API; drain instead.
    OSCMessage drained;
    while (queue.pop(drained))
    {
    }
    EXPECT_TRUE(queue.isEmpty());
}

// ========== OSCServer Tests ==========

class OSCServerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        server = std::make_unique<OSCServer>("127.0.0.1", 9001);
    }

    void TearDown() override
    {
        if (server)
        {
            server->stop();
        }
    }

    std::unique_ptr<OSCServer> server;
};

TEST_F(OSCServerTest, StartsAndStops)
{
    EXPECT_TRUE(server->start());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_NO_THROW(server->stop());
}

TEST_F(OSCServerTest, ReceivesMessage)
{
    ASSERT_TRUE(server->start());

    // Send message from client
    OSCClient client("127.0.0.1", 9001);
    OSCMessage msg;
    msg.setAddress("/hello");
    msg.addFloat(123.0f);
    EXPECT_TRUE(client.send(msg));

    // Poll server queue for up to 500ms
    OSCMessage received;
    bool got = false;
    for (int i = 0; i < 50; ++i)
    {
        if (server->getMessageQueue().pop(received))
        {
            got = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    server->stop();

    ASSERT_TRUE(got);
    EXPECT_EQ(received.getAddress(), "/hello");
    ASSERT_EQ(received.getArgumentCount(), 1u);
    EXPECT_FLOAT_EQ(std::get<float>(received.getArgument(0)), 123.0f);
}

// ========== OSCClient Tests ==========

class OSCClientTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        client = std::make_unique<OSCClient>("127.0.0.1", 9002);
    }

    void TearDown() override
    {
        // OSCClient has no start/stop in current API; cleanup is via destructor.
    }

    std::unique_ptr<OSCClient> client;
};

TEST_F(OSCClientTest, SendsMessage)
{
    // Stand up a local server to ensure send returns true.
    OSCServer server("127.0.0.1", 9002);
    ASSERT_TRUE(server.start());

    OSCMessage msg;
    msg.setAddress("/test");
    msg.addFloat(42.0f);

    EXPECT_TRUE(client->send(msg));

    server.stop();
}

// ========== OSCHub Tests ==========

class OSCHubTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        OSCHub::Config cfg;
        cfg.serverAddress = "127.0.0.1";
        cfg.serverPort = 9003;
        cfg.clientAddress = "127.0.0.1";
        cfg.clientPort = 9004;
        cfg.queueSize = 4096;
        hub = std::make_unique<OSCHub>(cfg);
    }

    void TearDown() override
    {
        if (hub)
        {
            hub->stop();
        }
    }

    std::unique_ptr<OSCHub> hub;
};

TEST_F(OSCHubTest, StartsAndStops)
{
    EXPECT_TRUE(hub->start());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_NO_THROW(hub->stop());
}

TEST_F(OSCHubTest, BidirectionalCommunication)
{
    ASSERT_TRUE(hub->start());

    // Create counterpart: server on 9004, client to 9003
    OSCServer remoteServer("127.0.0.1", 9004);
    OSCClient remoteClient("127.0.0.1", 9003);

    ASSERT_TRUE(remoteServer.start());

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Hub sends to remote
    OSCMessage toRemote;
    toRemote.setAddress("/to_remote");
    EXPECT_TRUE(hub->sendMessage(toRemote));

    // Remote sends to hub
    OSCMessage toHub;
    toHub.setAddress("/to_hub");
    EXPECT_TRUE(remoteClient.send(toHub));

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Verify remote received the hub message
    OSCMessage remoteReceived;
    bool gotRemote = false;
    for (int i = 0; i < 50; ++i)
    {
        if (remoteServer.getMessageQueue().pop(remoteReceived))
        {
            gotRemote = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Verify hub received the remote message
    OSCMessage hubReceived;
    bool gotHub = false;
    for (int i = 0; i < 50; ++i)
    {
        if (hub->receiveMessage(hubReceived))
        {
            gotHub = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    remoteServer.stop();
    hub->stop();

    ASSERT_TRUE(gotRemote);
    EXPECT_EQ(remoteReceived.getAddress(), "/to_remote");
    ASSERT_TRUE(gotHub);
    EXPECT_EQ(hubReceived.getAddress(), "/to_hub");
}

// ========== Performance Benchmarks ==========

class OSCPerformanceBenchmark : public ::testing::Test
{
protected:
    OSCClient client{"127.0.0.1", 9005};
    OSCMessage testMsg;

    void SetUp() override
    {
        testMsg.setAddress("/benchmark");
        testMsg.addFloat(1.0f);
        testMsg.addFloat(2.0f);
        testMsg.addFloat(3.0f);

        // Stand up receiver so sends can succeed.
        // NOTE: Benchmarks are disabled by default.
    }

    void TearDown() override
    {
    }
};

TEST_F(OSCPerformanceBenchmark, DISABLED_SendLatency)
{
    OSCServer server("127.0.0.1", 9005);
    ASSERT_TRUE(server.start());

    constexpr int iterations = 1000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        client.send(testMsg);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avgMicros = static_cast<double>(duration.count()) / iterations;

    std::cout << "Average OSC send time: " << avgMicros << " μs\n";

    EXPECT_LT(avgMicros, 100.0); // Target: <100μs per send

    server.stop();
}

TEST_F(OSCPerformanceBenchmark, DISABLED_MessageQueueThroughput)
{
    RTMessageQueue queue{10000};
    constexpr int iterations = 10000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i)
    {
        queue.push(testMsg);
    }

    auto pushEnd = std::chrono::high_resolution_clock::now();

    OSCMessage retrieved;
    for (int i = 0; i < iterations; ++i)
    {
        queue.pop(retrieved);
    }

    auto popEnd = std::chrono::high_resolution_clock::now();

    auto pushDuration = std::chrono::duration_cast<std::chrono::microseconds>(pushEnd - start);
    auto popDuration = std::chrono::duration_cast<std::chrono::microseconds>(popEnd - pushEnd);

    double avgPushMicros = static_cast<double>(pushDuration.count()) / iterations;
    double avgPopMicros = static_cast<double>(popDuration.count()) / iterations;

    std::cout << "Average queue push: " << avgPushMicros << " μs\n";
    std::cout << "Average queue pop: " << avgPopMicros << " μs\n";

    EXPECT_LT(avgPushMicros, 1.0); // Lock-free should be <1μs
    EXPECT_LT(avgPopMicros, 1.0);
}
