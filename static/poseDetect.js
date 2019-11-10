/**
 * Created by chad hart on 11/30/17.
 * Client side of Tensor Flow Object Detection Web API
 * Written for webrtcHacks - https://webrtchacks.com
 */

//Parameters
const s = document.getElementById('poseDetect');
const sourceVideo = s.getAttribute("data-source");  //the source video to use
const uploadWidth = s.getAttribute("data-uploadWidth") || 640; //the width of the upload file
const mirror = s.getAttribute("data-mirror") || false; //mirror the boundary boxes
const scoreThreshold = s.getAttribute("data-scoreThreshold") || 0.1;
const apiServer = s.getAttribute("data-apiServer") || window.location.origin + '/show-pose'; //the full TensorFlow Object Detection API server url

//Video element selector
v = document.getElementById(sourceVideo);

//for starting events
let isPlaying = false,
    gotMetadata = false;

//Canvas setup

//create a canvas to grab an image for upload
let imageCanvas = document.createElement('canvas');
let imageCtx = imageCanvas.getContext("2d");

//create a canvas for drawing object boundaries
let drawCanvas = document.createElement('canvas');
document.body.appendChild(drawCanvas);
let drawCtx = drawCanvas.getContext("2d");

const POSE_PAIRS = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            [1, 5],
            [5, 6],
            [6, 7],
            [1, 14],
            [14, 8],
            [8, 9],
            [9, 10],
            [14, 11],
            [11, 12],
            [12, 13],
        ];

//draw boxes and labels on each detected object
function drawPoses(points) {

    console.log(points);
    //clear the previous drawings
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    let i = 0;
    points.forEach(point =>{
        let confidence = parseFloat(Math.round(point.score * 100) / 100).toFixed(2);
        if (confidence > scoreThreshold){

            let circle = new Path2D();
            circle.arc(point.y, point.x, 10, 0, 2 * Math.PI);
            drawCtx.fill(circle);

            drawCtx.fillText([i, confidence, point.name], point.y+10, point.x);
            i++;
        }

    }); 

    // POSE_PAIRS.forEach(pose_pair =>{
    //     let point_a = points[pose_pair[0]];
    //     let point_b = points[pose_pair[1]];
    //
    //     if (point_a.confidence > scoreThreshold && point_b.confidence > scoreThreshold) {
    //         drawCtx.beginPath();
    //         drawCtx.moveTo(point_a.x, point_a.y);
    //         drawCtx.lineTo(point_b.x, point_b.y);
    //         drawCtx.stroke();
    //     }
    // });
    // console.log("points_drawn:", point_drawn);

}

//Add file blob to a form and post
function postFile(file) {

    posenet.load().then(net => {
        return net.estimateMultiplePoses(file, {
            flipHorizontal:false,
            maxDetections: 2,
            scoreThreshold: scoreThreshold,
            nmsRadius: 20
        })
    }).then(poses => {
        console.log(poses);
        let xhr = new XMLHttpRequest();
        xhr.open('POST', apiServer, true);
        xhr.setRequestHeader('Content-type', 'application/json');
        xhr.send(JSON.stringify({
            value: poses
        }));

        xhr.onload = function () {
            if (this.status == 200) {
                let objects = JSON.parse(this.response);
                drawPoses(objects);

                imageCtx.drawImage(v, 0, 0, v.videoWidth, v.videoHeight, 0, 0, uploadWidth, uploadWidth * (v.videoHeight / v.videoWidth));
                let dataUrl = imageCanvas.toDataURL("image/jpeg");
                let image = document.createElement("img");
                image.src = dataUrl;
                postFile(image);
                let el = document.getElementById('testdownload');
                el.href = dataUrl;
                // let show_img = document.getElementById("show-img");
                // show_img.src = dataUrl;
            } else {
                console.error(xhr);
            }
        }
    })
}

//Start object detection
function startObjectDetection() {

    console.log("starting object detection");

    //Set canvas sizes base don input video
    drawCanvas.width = v.videoWidth;
    drawCanvas.height = v.videoHeight;

    imageCanvas.width = uploadWidth;
    imageCanvas.height = uploadWidth * (v.videoHeight / v.videoWidth);

    //Some styles for the drawcanvas
    drawCtx.lineWidth = 4;
    drawCtx.strokeStyle = "cyan";
    drawCtx.fillStyle = "red";
    drawCtx.font = "bold 30px Courier New, sans-serif";

    //Save and send the first image
    imageCtx.drawImage(v, 0, 0, v.videoWidth, v.videoHeight, 0, 0, uploadWidth, uploadWidth * (v.videoHeight / v.videoWidth));
    // imageCanvas.toBlob(postFile, 'image/jpeg');
    let image = document.createElement("img");
    let dataUrl = imageCanvas.toDataURL("image/jpeg");
    image.src = dataUrl;
    postFile(image);
    let el = document.getElementById('testdownload');
    el.href = dataUrl;
    // let show_img = document.getElementById("show-img");
    // show_img.src = dataUrl;
}

//Starting events

//check if metadata is ready - we need the video size
v.onloadedmetadata = () => {
    console.log("video metadata ready");
    gotMetadata = true;
    if (isPlaying)
        startObjectDetection();
};

//see if the video has started playing
v.onplaying = () => {
    console.log("video playing");
    isPlaying = true;
    if (gotMetadata) {
        startObjectDetection();
    }
};

