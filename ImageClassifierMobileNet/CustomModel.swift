//
//  CustomModel.swift
//  ImageClassifierMobileNet
//
//  Created by Vishwas Prakash on 16/12/24.
//

import CoreML
import Vision
import AVFoundation
import CoreImage

class CameraFeedHandler: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var predictions: [Float] = [] // Example: Update based on your use case
    var captureSession: AVCaptureSession!
    private var model: converted_model!
    private var isProcessingFrame = false
    let predictionMap: [String] = ["indusind", "hdfc", "icici", "yesbank", "axis", "sbi", "pnb"]
    
    init(model: converted_model) {
        super.init()
        self.model = model
        setupCamera()
    }
    
    private func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .vga640x480
//        captureSession.sessionPreset = .low
        
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            fatalError("Unable to access the camera")
        }
        
        captureSession.addInput(input)
        
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "CameraFeedQueue"))
        captureSession.addOutput(videoOutput)
    }
    
    func startCamera() {
        captureSession.startRunning()
    }
    
    func stopCamera() {
        captureSession.stopRunning()
    }
    
    private func resizePixelBuffer(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
        var resizedPixelBuffer: CVPixelBuffer?
        let attributes = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attributes, &resizedPixelBuffer)
        guard let buffer = resizedPixelBuffer else { return nil }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        context.render(ciImage, to: buffer)
        return buffer
    }
    
    
    private func processFrame(_ pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        // Resize and normalize the frame
        guard let resizedBuffer = resizePixelBuffer(pixelBuffer, width: 224, height: 224),
              let array = try? MLMultiArray(shape: [1, 224, 224, 3], dataType: .float32) else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(resizedBuffer, .readOnly)
        let width = CVPixelBufferGetWidth(resizedBuffer)
        let height = CVPixelBufferGetHeight(resizedBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(resizedBuffer)
        guard let baseAddress = CVPixelBufferGetBaseAddress(resizedBuffer) else {
            CVPixelBufferUnlockBaseAddress(resizedBuffer, .readOnly)
            return nil
        }
        
        let bufferPointer = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * bytesPerRow + x * 4
                let r = Float(bufferPointer[offset + 1]) / 255.0
                let g = Float(bufferPointer[offset + 2]) / 255.0
                let b = Float(bufferPointer[offset + 3]) / 255.0
                
                array[[0, y, x, 0] as [NSNumber]] = r as NSNumber
                array[[0, y, x, 1] as [NSNumber]] = g as NSNumber
                array[[0, y, x, 2] as [NSNumber]] = b as NSNumber
            }
        }
        
        CVPixelBufferUnlockBaseAddress(resizedBuffer, .readOnly)
        return array
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard !isProcessingFrame else { return }
        isProcessingFrame = true
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let inputArray = processFrame(pixelBuffer) else {
            isProcessingFrame = false
            return
        }
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let input = converted_modelInput(input_1: inputArray)
                let outputFeatures = try self.model.prediction(input: input)
                let modelOutput = converted_modelOutput(features: outputFeatures)
                
                DispatchQueue.main.async {
                    self.predictions = self.multiArrayToFloatArray(modelOutput.Identity_1)
                    self.isProcessingFrame = false
                    
                }
            } catch {
                print("Prediction failed: \(error)")
                self.isProcessingFrame = false
            }
        }
    }
    
    func multiArrayToFloatArray(_ multiArray: MLMultiArray) -> [Float] {
        let count = multiArray.count
        var floatArray: [Float] = Array(repeating: 0, count: count)
        for i in 0..<count {
            floatArray[i] = Float(truncating: multiArray[i])
        }
        return floatArray
    }

}
