 
import SwiftUI
import AVFoundation
import Vision

struct ContentView: View {
    @StateObject private var cameraFeedHandler: CameraFeedHandler
        
        init() {
            let model = try! converted_model(configuration: MLModelConfiguration())
            _cameraFeedHandler = StateObject(wrappedValue: CameraFeedHandler(model: model))
        }
        
        var body: some View {
            ZStack {
                // Camera feed as the background
                CameraPreviewView(cameraFeedHandler: cameraFeedHandler)
                    .edgesIgnoringSafeArea(.all)
                
                // Align the VStack to the bottom
                VStack {
                    Spacer() // Push the content to the bottom
                    
                    VStack {
                        Text("Predictions:")
                            .font(.headline)
                            .foregroundColor(.white) // To ensure text is visible
                        
                        List(cameraFeedHandler.predictions.indices, id: \.self) { index in
                            HStack {
                                Text("\(cameraFeedHandler.predictionMap[index])")
                                    .foregroundColor(.white)
                                Spacer()
                                Text(String(format: "%.4f", cameraFeedHandler.predictions[index]))
                                    .foregroundColor(.white)
                            }
                        }
                        .listStyle(PlainListStyle()) // Adjust list style to fit design
                        .frame(height: 300) // Limit list height
                    }
                    .padding()
                    .background(Color.black.opacity(0.3)) // Semi-transparent black background
                    .cornerRadius(12) // Rounded corners
                    .frame(height: 350)
                    .padding(.horizontal, 16) // Add padding from the sides
                    .padding(.bottom, 30) // Padding from the bottom of the screen
                }
            }
            .onAppear {
                cameraFeedHandler.startCamera()
            }
            .onDisappear {
                cameraFeedHandler.stopCamera()
            }
        }
//    @StateObject private var viewModel = RealtimeClassificationViewModel()
//
//    var body: some View {
//        ZStack {
//            // Camera feed
//            CameraView(viewModel: viewModel)
//                .edgesIgnoringSafeArea(.all)
//
//            // Classification result overlay
//            if let result = viewModel.classificationResult {
//                VStack {
//                    Spacer() // Push text to the bottom
//                    Text("\(result.label) (\(String(format: "%.2f", result.confidence * 100))%)")
//                        .font(.title)
//                        .bold()
//                        .padding()
//                        .background(Color.black.opacity(0.7))
//                        .foregroundColor(.white)
//                        .clipShape(Capsule())
//                        .padding(.bottom, 50)
//                }
//            }
//        }
//        .onAppear {
//            viewModel.startCamera()
//        }
//        .onDisappear {
//            viewModel.stopCamera()
//        }
//    }
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

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

struct CameraPreviewView: UIViewControllerRepresentable {
    var cameraFeedHandler: CameraFeedHandler
    
    func makeUIViewController(context: Context) -> CameraPreviewViewController {
        return CameraPreviewViewController(cameraFeedHandler: cameraFeedHandler)
    }
    
    func updateUIViewController(_ uiViewController: CameraPreviewViewController, context: Context) {
        // No need to update the view controller
    }
}

class CameraPreviewViewController: UIViewController {
    private var cameraFeedHandler: CameraFeedHandler
    private var previewLayer: AVCaptureVideoPreviewLayer!
    
    init(cameraFeedHandler: CameraFeedHandler) {
        self.cameraFeedHandler = cameraFeedHandler
        super.init(nibName: nil, bundle: nil)
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        previewLayer = AVCaptureVideoPreviewLayer(session: cameraFeedHandler.captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        previewLayer.frame = view.bounds
    }
}
