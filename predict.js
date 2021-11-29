let imageLoaded = false;
$("#image-selector").change(function () {
	imageLoaded = false;
	let reader = new FileReader();
	reader.onload = function () {
		let dataURL = reader.result;
		$("#selected-image").attr("src", dataURL);
		$("#prediction-list").empty();
		imageLoaded = true;
	}
	
	let file = $("#image-selector").prop('files')[0];
	reader.readAsDataURL(file);
});

let binmodel;
let modelLoaded = false;
$( document ).ready(async function () {
	modelLoaded = false;
	$('.progress-bar').show();
    console.log( "Loading model..." );
    binmodel = await tf.loadLayersModel('model/binary.json');
	multimodel = await tf.loadLayersModel('model/multiclass_pneumonia.json');
    console.log( "Model loaded." );
	$('.progress-bar').hide();
	modelLoaded = true;
});

$("#predict-button").click(async function () {
	if (!modelLoaded) { alert("The model must be loaded first"); return; }
	if (!imageLoaded) { alert("Please select an image first"); return; }
	
	let image = $('#selected-image').get(0);
	
	// Pre-process the image
	console.log( "Loading image..." );
	let tensor = tf.browser.fromPixels(image, 1)
		.resizeNearestNeighbor([350,350]) // change the image size
		.expandDims()
		.toFloat()
		.div(tf.scalar(255.0))// RGB -> BGR
	let binpredict = await binmodel.predict(tensor).data();
	$('#prediction-list').empty()
	$('#prediction-list').append(`<li>pneumonia: ${binpredict[0].toFixed(6)}</li>`)

	let multipredicts = await multimodel.predict(tensor).data();
	console.log(multipredicts)
	let top5 = Array.from(multipredicts)
		.map(function (p, i) { // this is Array.map
			return {
				probability: p,
				className: MULTICLASS[i] // we are selecting the value from the obj
			};
		}).sort(function (a, b) {
			return b.probability - a.probability;
		}).slice(0, 2);
	top5.forEach(function (p) {
		$("#prediction-list").append(`<li>${p.className}: ${p.probability.toFixed(6)}</li>`);
	});
});
