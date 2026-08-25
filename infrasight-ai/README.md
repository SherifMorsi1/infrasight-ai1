# InfraSight AI

**Browser-based computer vision for transportation, aviation, and smart infrastructure.**

InfraSight AI runs object detection directly in the browser using TensorFlow.js and the COCO-SSD model. Users can upload an image or enable a webcam and immediately receive bounding boxes, confidence scores, and scene-level counts for vehicles, people, aircraft, and other detected objects.

The project is intentionally deployment-light: it is a static site with no backend, no database, no build step, and no local machine-learning environment required for the live demo.

## Features

- Deep-learning object detection with COCO-SSD and a MobileNet v2 backbone
- Image upload and live webcam inference
- Bounding boxes, class labels, and confidence scores rendered with the Canvas API
- Transportation-focused scene analytics for vehicles, people, and aircraft
- Configurable confidence threshold
- Responsive interface for desktop and mobile browsers
- Client-side application processing
- Dependency-free unit tests for analytics logic
- GitHub Actions CI
- GitHub Pages-compatible static deployment

## Live architecture

```text
Image / Webcam
      |
      v
TensorFlow.js
      |
      v
COCO-SSD (MobileNet v2)
      |
      v
Object detections
      |
      +--> Bounding boxes + labels
      |
      +--> Scene aggregation
               |
               v
        Dashboard metrics
```

## Run the demo

The repository is designed for GitHub Pages. Once Pages is enabled for the repository, the application can be opened directly from the Pages URL without installing anything.

For optional local development, any static web server can serve the repository root. This is not required for the deployed demo.

## GitHub Pages deployment

1. Push the repository to GitHub.
2. Open **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Select the `main` branch and `/ (root)` folder.
5. Save. GitHub will publish the static application.

## Technology

- JavaScript ES modules
- TensorFlow.js 4.22.0
- TensorFlow.js COCO-SSD 2.2.2
- HTML5 Canvas API
- MediaDevices / WebRTC camera access
- HTML5 + CSS3
- Node.js built-in test runner for unit tests
- GitHub Actions
- GitHub Pages

## Testing

The project contains pure-function unit tests for detection aggregation and display formatting.

```bash
npm test
npm run check
```

GitHub Actions runs both checks on every push and pull request.

## Model and limitations

InfraSight AI uses the pretrained TensorFlow.js COCO-SSD object detector rather than a custom-trained model. COCO-SSD detects objects from the COCO class set and is suitable for demonstrating browser-side deep-learning inference. It is not a safety system and should not be used for operational, security, or autonomous-control decisions.

Detection quality depends on image quality, viewpoint, lighting, object scale, and whether the object belongs to a supported COCO class. Scene counts are descriptive summaries of model detections, not validated traffic or safety metrics.

## Privacy

Images and webcam frames are processed by the application in the user's browser. The application does not implement an upload API or backend media store. TensorFlow.js and model assets are loaded from external CDNs/model hosting when the page starts.

## Attribution

This project uses TensorFlow.js and the TensorFlow.js COCO-SSD model from the TensorFlow project. Those dependencies are loaded at runtime and retain their respective licenses.

## License

Application code in this repository is available under the MIT License.
