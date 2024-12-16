 
import SwiftUI
import AVFoundation
import Vision

struct ContentView: View {
    @StateObject private var viewModel = RealtimeClassificationViewModel()

    var body: some View {
//        VStack{
//            CameraFeed()
//                .edgesIgnoringSafeArea(.all)
//        }
        ZStack {
            // Camera feed
            CameraView(viewModel: viewModel)
                .edgesIgnoringSafeArea(.all)

            // Classification result overlay
            if let result = viewModel.classificationResult {
                VStack {
                    Spacer() // Push text to the bottom
                    Text("\(result.label) (\(String(format: "%.2f", result.confidence * 100))%)")
                        .font(.title)
                        .bold()
                        .padding()
                        .background(Color.black.opacity(0.7))
                        .foregroundColor(.white)
                        .clipShape(Capsule())
                        .padding(.bottom, 50)
                }
            }
        }
        .onAppear {
            viewModel.startCamera()
        }
        .onDisappear {
            viewModel.stopCamera()
        }
    }
}

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

struct CameraView: UIViewRepresentable {
    let viewModel: RealtimeClassificationViewModel
    private let previewLayer = AVCaptureVideoPreviewLayer()

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        
        previewLayer.session = viewModel.captureSession
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
//        context.coordinator.previewLayer.videoGravity = .resizeAspectFill
//        view.layer.addSublayer(context.coordinator.previewLayer)
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        // Ensure the preview layer matches the size of the view
        DispatchQueue.main.async {
            self.previewLayer.frame = uiView.bounds
        }
    }

//    func makeCoordinator() -> Coordinator {
//        return Coordinator(viewModel: viewModel)
//    }

//    class Coordinator: NSObject {
//        let previewLayer: AVCaptureVideoPreviewLayer
//
//        init(viewModel: RealtimeClassificationViewModel) {
////            self.previewLayer = AVCaptureVideoPreviewLayer(session: viewModel.captureSession)
//            self.previewLayer.session = viewModel.captureSession
//            print("View model got initialized")
//        }
//    }
}

struct CameraFeed: UIViewRepresentable{
    private let captureSession = AVCaptureSession()
    private let previewLayer = AVCaptureVideoPreviewLayer()

    func makeUIView(context: Context) -> UIView {
        let view = UIView()

        // Configure the camera session
        captureSession.sessionPreset = .high

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: camera) else {
            print("Failed to access the camera.")
            return view
        }

        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        }

        // Configure the preview layer
        previewLayer.session = captureSession
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)

        // Start the camera session
        DispatchQueue.global(qos: .userInitiated).async {
            self.captureSession.startRunning()
        }

        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        // Ensure the preview layer matches the size of the view
        DispatchQueue.main.async {
            self.previewLayer.frame = uiView.bounds
        }
    }

    func dismantleUIView(_ uiView: UIView, context: Context) {
        captureSession.stopRunning()
    }
}


struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
