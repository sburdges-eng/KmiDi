#include "main_window.h"
#include <QMenuBar>
#include <QToolBar>
#include <QStatusBar>
#include <QLabel>
#include <QVBoxLayout>

namespace kelly {

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent) {
    setupUi();
    createMenus();
    createToolbar();

    statusBar()->showMessage("Ready");
}

void MainWindow::setupUi() {
    // Central widget setup
    auto *centralWidget = new QWidget(this);
    auto *layout = new QVBoxLayout(centralWidget);

    // Add a welcome label to show the app is working
    auto *welcomeLabel = new QLabel("KmiDi - Unified Music Intelligence & Audio Workstation", centralWidget);
    welcomeLabel->setAlignment(Qt::AlignCenter);
    welcomeLabel->setStyleSheet("font-size: 24px; font-weight: bold; padding: 20px;");
    layout->addWidget(welcomeLabel);

    // Add a status label
    auto *statusLabel = new QLabel("Application is running. UI components will be added here.", centralWidget);
    statusLabel->setAlignment(Qt::AlignCenter);
    statusLabel->setStyleSheet("font-size: 14px; color: #666; padding: 10px;");
    layout->addWidget(statusLabel);

    centralWidget->setLayout(layout);
    setCentralWidget(centralWidget);
}

void MainWindow::createMenus() {
    auto *fileMenu = menuBar()->addMenu("&File");
    auto *editMenu = menuBar()->addMenu("&Edit");
    auto *helpMenu = menuBar()->addMenu("&Help");

    // Actions would be added here
}

void MainWindow::createToolbar() {
    auto *toolbar = addToolBar("Main");
    // Toolbar actions would be added here
}

} // namespace kelly
