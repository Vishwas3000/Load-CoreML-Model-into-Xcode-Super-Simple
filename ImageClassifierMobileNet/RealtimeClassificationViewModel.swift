//
//  RealtimeClassificationViewModel.swift
//  ImageClassifierMobileNet
//
//  Created by Vishwas Prakash on 16/12/24.
//

import Foundation
import AVFoundation
import Vision

class RealtimeClassificationViewModel: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var classificationResult: (label: String, confidence: Double)?
    public let captureSession = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let model: VNCoreMLModel
    private let queue = DispatchQueue(label: "camera.frame.processing.queue")

    override init() {
        // Load Core ML model
        guard let modelURL = Bundle.main.url(forResource: "MobileNetV2", withExtension: "mlmodelc"),
              let loadedModel = try? MLModel(contentsOf: modelURL) else {
            fatalError("Failed to load MobileNetV2 model.")
        }
        self.model = try! VNCoreMLModel(for: loadedModel)

        super.init()
        setupCaptureSession()
    }

    private func setupCaptureSession() {
        captureSession.sessionPreset = .vga640x480 // Optimize for speed, lower resolution
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            fatalError("Unable to access camera.")
        }

        // Add camera input
        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        }

        // Configure video output
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
    }

    func startCamera() {
        if !captureSession.isRunning {
            captureSession.startRunning()
        }
    }

    func stopCamera() {
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
    }

    // Delegate method to process video frames
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        // Perform prediction
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let results = request.results as? [VNClassificationObservation], let topResult = results.first else {
                return
            }
            DispatchQueue.main.async {
                self?.classificationResult = (label: topResult.identifier, confidence: Double(topResult.confidence))
            }
        }

        request.usesCPUOnly = false // Utilize GPU for better performance

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("Failed to perform classification: \(error)")
        }
    }
}
