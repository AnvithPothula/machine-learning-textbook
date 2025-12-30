// K Selection Interactive Simulator MicroSim
// Shows how different k values affect KNN decision boundaries and predictions

// Canvas dimensions
let canvasWidth = 800;
let drawHeight = 600;
let controlHeight = 110;
let canvasHeight = drawHeight + controlHeight;
let margin = 40;
let sliderLeftMargin = 180;
let defaultTextSize = 16;

// Data and parameters
let trainingData = [];
let testPoint;
let kValue = 5;
let showVoronoi = false;
let dragging = false;

// Colors for 3 classes
let classColors = [];

// Controls
let kSlider;
let voronoiCheckbox;
let addNoiseBtn;
let resetBtn;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.parent(document.querySelector('main'));

  // Define class colors
  classColors = [
    color(33, 150, 243),   // Blue
    color(255, 152, 0),    // Orange
    color(76, 175, 80)     // Green
  ];

  // Initialize test point at center
  testPoint = createVector(canvasWidth / 2, drawHeight / 2);

  // Generate initial training data
  generateTrainingData();

  // Create k slider
  kSlider = createSlider(1, 25, kValue, 1);
  kSlider.position(sliderLeftMargin, drawHeight + 15);
  kSlider.size(canvasWidth - sliderLeftMargin - margin);

  // Create voronoi checkbox
  voronoiCheckbox = createCheckbox('Show Voronoi (k=1)', showVoronoi);
  voronoiCheckbox.position(10, drawHeight + 50);
  voronoiCheckbox.changed(() => {
    showVoronoi = voronoiCheckbox.checked();
  });

  // Create add noise button
  addNoiseBtn = createButton('Add Noise');
  addNoiseBtn.position(200, drawHeight + 48);
  addNoiseBtn.mousePressed(addNoise);
  addNoiseBtn.size(100, 25);

  // Create reset button
  resetBtn = createButton('Reset');
  resetBtn.position(310, drawHeight + 48);
  resetBtn.mousePressed(resetData);
  resetBtn.size(80, 25);

  describe('Interactive KNN visualization showing how k value affects decision boundaries and predictions', LABEL);
}

function draw() {
  updateCanvasSize();
  kValue = kSlider.value();

  // Drawing area background
  fill('aliceblue');
  stroke('silver');
  rect(0, 0, canvasWidth, drawHeight);

  // Control area background
  fill('white');
  noStroke();
  rect(0, drawHeight, canvasWidth, controlHeight);

  // Draw title
  fill('black');
  textSize(20);
  textAlign(CENTER, TOP);
  text('K-Nearest Neighbors: Effect of k Value', canvasWidth / 2, 10);

  // Reset text settings
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);

  // Draw decision boundary background
  if (showVoronoi && kValue === 1) {
    drawVoronoiBackground();
  } else {
    drawDecisionBackground();
  }

  // Draw training data points
  drawTrainingData();

  // Draw test point and its k-nearest neighbors
  drawKNearestNeighbors();
  drawTestPoint();

  // Draw info panel
  drawInfoPanel();

  // Draw control labels
  fill('black');
  noStroke();
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('k value: ' + kValue, 10, drawHeight + 20);

  // Draw instruction
  fill(100);
  textSize(13);
  text('Drag the yellow test point to explore predictions', 10, drawHeight + 85);
}

function drawDecisionBackground() {
  // Draw simplified decision regions as a grid
  let step = 15;
  for (let x = margin; x < canvasWidth - margin; x += step) {
    for (let y = 40; y < drawHeight - margin; y += step) {
      let predictedClass = predictClass(x, y, kValue);
      let c = classColors[predictedClass];
      fill(red(c), green(c), blue(c), 30);
      noStroke();
      rect(x, y, step, step);
    }
  }
}

function drawVoronoiBackground() {
  // For k=1, show Voronoi diagram
  let step = 10;
  for (let x = margin; x < canvasWidth - margin; x += step) {
    for (let y = 40; y < drawHeight - margin; y += step) {
      let nearest = findNearestPoint(x, y);
      if (nearest) {
        let c = classColors[nearest.class];
        fill(red(c), green(c), blue(c), 40);
        noStroke();
        rect(x, y, step, step);
      }
    }
  }
}

function drawTrainingData() {
  for (let pt of trainingData) {
    fill(classColors[pt.class]);
    stroke(255);
    strokeWeight(1);
    circle(pt.x, pt.y, 12);
  }
}

function drawTestPoint() {
  // Draw test point
  fill(255, 235, 59); // Yellow
  stroke(0);
  strokeWeight(3);
  circle(testPoint.x, testPoint.y, 24);

  // Draw crosshair
  stroke(0);
  strokeWeight(2);
  line(testPoint.x - 12, testPoint.y, testPoint.x + 12, testPoint.y);
  line(testPoint.x, testPoint.y - 12, testPoint.x, testPoint.y + 12);
}

function drawKNearestNeighbors() {
  // Find k-nearest neighbors
  let neighbors = findKNearest(testPoint.x, testPoint.y, kValue);

  // Draw lines to k-nearest neighbors
  stroke(100);
  strokeWeight(1);
  drawingContext.setLineDash([3, 3]);
  for (let n of neighbors) {
    line(testPoint.x, testPoint.y, n.x, n.y);
  }
  drawingContext.setLineDash([]);

  // Highlight k-nearest neighbors
  for (let n of neighbors) {
    fill(classColors[n.class]);
    stroke(0);
    strokeWeight(3);
    circle(n.x, n.y, 16);
  }
}

function drawInfoPanel() {
  let prediction = predictClass(testPoint.x, testPoint.y, kValue);
  let neighbors = findKNearest(testPoint.x, testPoint.y, kValue);

  // Count votes
  let votes = [0, 0, 0];
  for (let n of neighbors) {
    votes[n.class]++;
  }

  // Draw info panel
  let panelX = canvasWidth - 240;
  let panelY = 45;
  let panelW = 230;
  let panelH = 140;

  fill(255, 255, 255, 240);
  stroke(200);
  strokeWeight(1);
  rect(panelX, panelY, panelW, panelH, 10);

  fill('black');
  noStroke();
  textAlign(LEFT, TOP);
  textSize(14);
  text('Test Point Prediction:', panelX + 10, panelY + 8);

  // Show prediction
  fill(classColors[prediction]);
  textSize(18);
  textStyle(BOLD);
  text('Class ' + prediction, panelX + 10, panelY + 30);
  textStyle(NORMAL);

  // Show vote breakdown
  fill('black');
  textSize(13);
  text('Neighbor votes (k=' + kValue + '):', panelX + 10, panelY + 60);

  for (let i = 0; i < 3; i++) {
    fill(classColors[i]);
    circle(panelX + 20, panelY + 85 + i * 20, 12);
    fill('black');
    textSize(13);
    text('Class ' + i + ': ' + votes[i] + ' vote' + (votes[i] !== 1 ? 's' : ''), panelX + 35, panelY + 82 + i * 20);
  }
}

function generateTrainingData() {
  trainingData = [];
  randomSeed(42); // Consistent data

  // Class 0 (Blue) - upper left
  for (let i = 0; i < 20; i++) {
    trainingData.push({
      x: randomGaussian(150, 35),
      y: randomGaussian(150, 35),
      class: 0
    });
  }

  // Class 1 (Orange) - upper right
  for (let i = 0; i < 20; i++) {
    trainingData.push({
      x: randomGaussian(canvasWidth - 150, 35),
      y: randomGaussian(150, 35),
      class: 1
    });
  }

  // Class 2 (Green) - bottom center
  for (let i = 0; i < 20; i++) {
    trainingData.push({
      x: randomGaussian(canvasWidth / 2, 40),
      y: randomGaussian(drawHeight - 120, 35),
      class: 2
    });
  }
}

function findKNearest(x, y, k) {
  // Calculate distances to all training points
  let distances = trainingData.map(pt => ({
    ...pt,
    dist: dist(x, y, pt.x, pt.y)
  }));

  // Sort by distance and take k nearest
  distances.sort((a, b) => a.dist - b.dist);
  return distances.slice(0, k);
}

function findNearestPoint(x, y) {
  if (trainingData.length === 0) return null;

  let nearest = trainingData[0];
  let minDist = dist(x, y, nearest.x, nearest.y);

  for (let pt of trainingData) {
    let d = dist(x, y, pt.x, pt.y);
    if (d < minDist) {
      minDist = d;
      nearest = pt;
    }
  }
  return nearest;
}

function predictClass(x, y, k) {
  let neighbors = findKNearest(x, y, k);

  // Count votes
  let votes = [0, 0, 0];
  for (let n of neighbors) {
    votes[n.class]++;
  }

  // Return class with most votes
  let maxVotes = Math.max(...votes);
  return votes.indexOf(maxVotes);
}

function addNoise() {
  // Add a random noise point
  trainingData.push({
    x: random(margin, canvasWidth - margin),
    y: random(60, drawHeight - margin),
    class: floor(random(3))
  });
}

function resetData() {
  generateTrainingData();
}

function mousePressed() {
  // Check if clicking on test point
  let d = dist(mouseX, mouseY, testPoint.x, testPoint.y);
  if (d < 20 && mouseY < drawHeight) {
    dragging = true;
  }
}

function mouseDragged() {
  if (dragging && mouseY < drawHeight) {
    testPoint.x = constrain(mouseX, margin, canvasWidth - margin);
    testPoint.y = constrain(mouseY, 60, drawHeight - margin);
  }
}

function mouseReleased() {
  dragging = false;
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(canvasWidth, canvasHeight);
  generateTrainingData(); // Regenerate with new positions
}

function updateCanvasSize() {
  const container = document.querySelector('main');
  if (container) {
    canvasWidth = container.offsetWidth;
    if (typeof kSlider !== 'undefined') {
      kSlider.size(canvasWidth - sliderLeftMargin - margin);
    }
  }
}
