const s = document.getElementById('poseDetect');
const sourceVideo = s.getAttribute("data-source");  //the source video to use
const scoreThreshold = s.getAttribute("data-scoreThreshold") || 0.1;
const apiServer = s.getAttribute("data-apiServer") || window.location.origin + '/show-pose'; //the full TensorFlow Object Detection API server url
var imageElement = document.getElementById('test_image');

const videoWidth = 600;
const videoHeight = 500;

//Video element selector
v = document.getElementById(sourceVideo);

posenet.load().then(function(net){
  return net.estimateMultiplePoses(imageElement, {
    flipHorizontal: false,
    maxDetections: 2,
    scoreThreshold: scoreThreshold,
    nmsRadius: 20})
}).then(function(poses){
  console.log(poses)
  let xhr = new XMLHttpRequest();
  xhr.open('POST', apiServer, true);
  xhr.setRequestHeader('Content-Type', 'application/json')
    xhr.send(JSON.stringify({
        value: poses
    }))
})
