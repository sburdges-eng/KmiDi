/*
 Core ML LLM Runner — stateful KV-cache greedy decode loop.

 Loads a compiled mlmodelc (from ExecuTorch export with --coreml-enable-state,
 --use_kv_cache, etc.), threads state output at step t back in as state input
 at step t+1, and decodes greedily. Optional: report ms/token for latency checks.

 Usage:
   .build/release/CoreMLLMRunner /path/to/Model.mlmodelc
   .build/release/CoreMLLMRunner /path/to/Model.mlmodelc --max-tokens 64 --timing

 Requires: macOS 15+ / iOS 18+ (stateful prediction API).
 */

import Foundation
import CoreML

// MARK: - Model discovery

func findStateInputOutputNames(_ model: MLModel) -> (stateIn: [String], stateOut: [String], tokenIn: String?, logitsOut: String?) {
    let desc = model.modelDescription
    var stateIn: [String] = []
    var stateOut: [String] = []
    var tokenIn: String?
    var logitsOut: String?

    for (name, _) in desc.inputDescriptionsByName {
        if name.lowercased().contains("state") || name.lowercased().contains("cache") || name.lowercased().contains("kv") {
            stateIn.append(name)
        } else if tokenIn == nil && (name.lowercased().contains("token") || name.lowercased().contains("input") || name.lowercased().contains("prompt")) {
            tokenIn = name
        }
    }
    if tokenIn == nil, let first = desc.inputDescriptionsByName.keys.first(where: { !$0.lowercased().contains("state") && !$0.lowercased().contains("cache") }) {
        tokenIn = first
    }

    for (name, _) in desc.outputDescriptionsByName {
        if name.lowercased().contains("state") || name.lowercased().contains("cache") || name.lowercased().contains("kv") {
            stateOut.append(name)
        } else if logitsOut == nil && (name.lowercased().contains("logits") || name.lowercased().contains("output")) {
            logitsOut = name
        }
    }
    if logitsOut == nil, let first = desc.outputDescriptionsByName.keys.first(where: { !$0.lowercased().contains("state") && !$0.lowercased().contains("cache") }) {
        logitsOut = first
    }

    return (stateIn, stateOut, tokenIn, logitsOut)
}

// MARK: - Runner

func run(
    modelURL: URL,
    maxTokens: Int,
    reportTiming: Bool
) throws {
    let config = MLModelConfiguration()
    config.computeUnits = .all

    let model = try MLModel(contentsOf: modelURL, configuration: config)
    let (stateInNames, stateOutNames, tokenInName, logitsOutName) = findStateInputOutputNames(model)

    guard let tokenInName = tokenInName, let logitsOutName = logitsOutName else {
        fputs("Could not identify token input and logits output. Inspect model I/O in Xcode.\n", stderr)
        exit(1)
    }

    // Placeholder token ID for first step (real usage: use tokenizer to get BOS)
    let bosTokenId: Int64 = 1
    var currentToken = bosTokenId
    // stateInputName -> MLFeatureValue to feed next step (from previous step's state outputs)
    var stateValues: [String: MLFeatureValue] = [:]
    var totalTime: TimeInterval = 0
    var tokenCount = 0

    for step in 0..<maxTokens {
        let start = CFAbsoluteTimeGetCurrent()

        var inputDict: [String: MLFeatureValue] = [:]

        // Token input (shape typically [1, 1] or [batch, seq=1])
        let tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
        tokenArray[0] = NSNumber(value: currentToken)
        inputDict[tokenInName] = MLFeatureValue(multiArray: tokenArray)

        // State inputs: feed previous step's state outputs (match by index: stateOut[i] -> stateIn[i])
        for (idx, inName) in stateInNames.enumerated() {
            if idx < stateOutNames.count, let val = stateValues[inName] {
                inputDict[inName] = val
            }
        }

        let inputs = try MLDictionaryFeatureProvider(dictionary: inputDict)
        let out = try model.prediction(from: inputs)

        totalTime += CFAbsoluteTimeGetCurrent() - start
        tokenCount += 1

        // Store state outputs for next step (stateOut[i] -> stateIn[i] for next iteration)
        for (idx, outName) in stateOutNames.enumerated() {
            if let f = out.featureValue(for: outName), idx < stateInNames.count {
                stateValues[stateInNames[idx]] = f
            }
        }

        // Logits: shape [1, 1, vocab_size] or [1, vocab_size]; take argmax for greedy
        guard let logitsFeature = out.featureValue(for: logitsOutName),
              let logits = logitsFeature.multiArrayValue else {
            break
        }
        let vocabSize: Int = logits.shape.count >= 2 ? logits.shape[logits.shape.count - 1].intValue : logits.count
        var maxIdx = 0
        var maxVal: Float = -Float.infinity
        let ptr = logits.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<vocabSize {
            let v = ptr[i]
            if v > maxVal {
                maxVal = v
                maxIdx = i
            }
        }
        currentToken = Int64(maxIdx)

        if reportTiming {
            let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000
            fputs(String(format: "step %d token %d ms %.2f\n", step, maxIdx, ms), stderr)
        }

        // EOS = 2 often; stop if you have a real EOS id from the tokenizer
        if currentToken == 2 {
            break
        }
    }

    if reportTiming && tokenCount > 0 {
        let avgMs = (totalTime * 1000) / Double(tokenCount)
        fputs(String(format: "Total: %d tokens, %.2f ms/token avg\n", tokenCount, avgMs), stderr)
    }
}

// MARK: - Entry

let args = CommandLine.arguments.dropFirst()
guard let modelPath = args.first else {
    fputs("Usage: CoreMLLMRunner <path-to-mlmodelc> [--max-tokens N] [--timing]\n", stderr)
    exit(1)
}

var maxTokens = 32
var timing = false
var i = args.index(after: args.startIndex)
while i < args.endIndex {
    if args[i] == "--max-tokens", i + 1 < args.endIndex {
        maxTokens = Int(args[i + 1]) ?? 32
        i += 2
        continue
    }
    if args[i] == "--timing" {
        timing = true
        i += 1
        continue
    }
    i += 1
}

let url = URL(fileURLWithPath: modelPath)
guard FileManager.default.fileExists(atPath: url.path) else {
    fputs("Model path does not exist: \(modelPath)\n", stderr)
    exit(1)
}

do {
    try run(modelURL: url, maxTokens: maxTokens, reportTiming: timing)
} catch {
    fputs("Error: \(error)\n", stderr)
    exit(1)
}
