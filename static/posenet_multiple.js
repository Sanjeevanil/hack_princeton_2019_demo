const s = document.getElementById('poseDetect');
const scoreThreshold = s.getAttribute("data-scoreThreshold") || 0.1;
const apiServer = s.getAttribute("data-apiServer") || window.location.origin + '/save-pose'; //the full TensorFlow Object Detection API server url

let images = document.images;

for (let image of images) {
  posenet.load().then(function(net){
    return net.estimateMultiplePoses(image, {
      flipHorizontal: false,
      maxDetections: 2,
      scoreThreshold: scoreThreshold,
      nmsRadius: 20})
  }).then(function(poses){
    let xhr = new XMLHttpRequest();
    xhr.open('POST', apiServer, true);
    xhr.setRequestHeader('Content-Type', 'application/json');
      xhr.send(JSON.stringify({
          value: poses,
          src: image.src
      }))
  });
}


