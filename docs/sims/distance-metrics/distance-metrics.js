// Distance Metrics Visualization MicroSim
// Comparing Euclidean vs Manhattan distance

// Canvas dimensions
let canvasWidth = 800;
let drawHeight = 600;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let margin = 40;
let defaultTextSize = 16;

// Points
let pointA = {x: 100, y: 150};
let pointB = {x: 300, y: 450};
let dragging = false;
let gridSpacing = 50;

// UI
let resetButton;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.parent(document.querySelector('main'));

  resetButton = createButton('Reset Point B');
  resetButton.position(10, drawHeight + 15);
  resetButton.mousePressed(resetPoints);
  resetButton.size(120, 25);

  describe('Interactive visualization comparing Euclidean and Manhattan distance metrics', LABEL);
}

function draw() {
  updateCanvasSize();

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
  text('Distance Metrics Comparison', canvasWidth / 2, 10);

  // Reset text settings
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);

  // Draw both visualizations side by side
  let halfWidth = (canvasWidth - 20) / 2;
  drawEuclideanSide(10, 50, halfWidth, drawHeight - 130);
  drawManhattanSide(halfWidth + 10, 50, halfWidth, drawHeight - 130);

  // Draw divider
  stroke(150);
  strokeWeight(2);
  line(canvasWidth / 2, 50, canvasWidth / 2, drawHeight - 80);

  // Calculate distances
  let euclidean = calculateEuclidean(pointA, pointB);
  let manhattan = calculateManhattan(pointA, pointB);
  let ratio = manhattan / euclidean;

  // Display comparison metrics at bottom
  fill('black');
  noStroke();
  textSize(14);
  textAlign(LEFT, CENTER);
  text(`Point A: (${Math.round(pointA.x)}, ${Math.round(pointA.y)})`, 10, drawHeight - 60);
  text(`Point B: (${Math.round(pointB.x)}, ${Math.round(pointB.y)})`, 10, drawHeight - 40);
  text(`Ratio (Manhattan/Euclidean): ${ratio.toFixed(3)}`, 10, drawHeight - 20);

  if (abs(ratio - sqrt(2)) < 0.05) {
    fill(156, 39, 176);
    textAlign(LEFT);
    text('⬅ Point B is on a diagonal! (ratio ≈ √2 ≈ 1.414)', 280, drawHeight - 20);
  }

  // Control labels
  fill('black');
  noStroke();
  textAlign(LEFT, CENTER);
  textSize(13);
  text('Drag the blue point B to compare distances', 150, drawHeight + 20);
}

function drawEuclideanSide(x, y, w, h) {
  push();
  translate(x, y);

  // Draw grid
  stroke(224);
  strokeWeight(1);
  for (let gx = 0; gx < w; gx += gridSpacing) {
    line(gx, 0, gx, h);
  }
  for (let gy = 0; gy < h; gy += gridSpacing) {
    line(0, gy, w, gy);
  }

  // Title
  fill(76, 175, 80);
  noStroke();
  textSize(16);
  textAlign(CENTER, TOP);
  text('Euclidean Distance', w / 2, 5);
  text('(Straight Line)', w / 2, 25);

  // Formula
  textSize(12);
  fill(100);
  text('d = √((x₂-x₁)² + (y₂-y₁)²)', w / 2, 45);

  // Map points to this panel
  let pAx = map(pointA.x, 0, canvasWidth, 0, w);
  let pAy = map(pointA.y, 50, drawHeight - 80, 0, h);
  let pBx = map(pointB.x, 0, canvasWidth, 0, w);
  let pBy = map(pointB.y, 50, drawHeight - 80, 0, h);

  // Draw straight line from A to B
  stroke(76, 175, 80);
  strokeWeight(3);
  line(pAx, pAy, pBx, pBy);

  // Draw points
  drawPoint(pAx, pAy, color(244, 67, 54), 'A');
  drawPoint(pBx, pBy, color(33, 150, 243), 'B');

  // Display distance
  let d = calculateEuclidean(pointA, pointB);
  fill(76, 175, 80);
  noStroke();
  textSize(16);
  textStyle(BOLD);
  text(`Distance: ${d.toFixed(1)}`, w / 2, h - 25);
  textStyle(NORMAL);

  pop();
}

function drawManhattanSide(x, y, w, h) {
  push();
  translate(x, y);

  // Draw grid
  stroke(224);
  strokeWeight(1);
  for (let gx = 0; gx < w; gx += gridSpacing) {
    line(gx, 0, gx, h);
  }
  for (let gy = 0; gy < h; gy += gridSpacing) {
    line(0, gy, w, gy);
  }

  // Title
  fill(255, 152, 0);
  noStroke();
  textSize(16);
  textAlign(CENTER, TOP);
  text('Manhattan Distance', w / 2, 5);
  text('(Grid Path)', w / 2, 25);

  // Formula
  textSize(12);
  fill(100);
  text('d = |x₂-x₁| + |y₂-y₁|', w / 2, 45);

  // Map points to this panel
  let pAx = map(pointA.x, 0, canvasWidth, 0, w);
  let pAy = map(pointA.y, 50, drawHeight - 80, 0, h);
  let pBx = map(pointB.x, 0, canvasWidth, 0, w);
  let pBy = map(pointB.y, 50, drawHeight - 80, 0, h);

  // Draw L-shaped path from A to B
  stroke(255, 152, 0);
  strokeWeight(3);
  line(pAx, pAy, pBx, pAy); // Horizontal
  line(pBx, pAy, pBx, pBy); // Vertical

  // Draw small circles at corner
  fill(255, 152, 0);
  noStroke();
  circle(pBx, pAy, 8);

  // Draw points
  drawPoint(pAx, pAy, color(244, 67, 54), 'A');
  drawPoint(pBx, pBy, color(33, 150, 243), 'B');

  // Display distance
  let d = calculateManhattan(pointA, pointB);
  fill(255, 152, 0);
  noStroke();
  textSize(16);
  textStyle(BOLD);
  text(`Distance: ${d.toFixed(1)}`, w / 2, h - 25);
  textStyle(NORMAL);

  pop();
}

function drawPoint(x, y, col, label) {
  fill(col);
  stroke(255);
  strokeWeight(2);
  circle(x, y, 24);
  fill(255);
  noStroke();
  textSize(14);
  textAlign(CENTER, CENTER);
  textStyle(BOLD);
  text(label, x, y);
  textStyle(NORMAL);
}

function calculateEuclidean(p1, p2) {
  return sqrt(pow(p2.x - p1.x, 2) + pow(p2.y - p1.y, 2));
}

function calculateManhattan(p1, p2) {
  return abs(p2.x - p1.x) + abs(p2.y - p1.y);
}

function mousePressed() {
  if (mouseY > drawHeight) return; // Don't drag in control area

  // Check if mouse is near point B
  let d = dist(mouseX, mouseY, pointB.x, pointB.y);
  if (d < 30) {
    dragging = true;
  }
}

function mouseDragged() {
  if (dragging && mouseY < drawHeight - 80) {
    pointB.x = constrain(mouseX, 50, canvasWidth - 50);
    pointB.y = constrain(mouseY, 100, drawHeight - 100);
  }
}

function mouseReleased() {
  dragging = false;
}

function resetPoints() {
  pointB = {x: 300, y: 450};
  dragging = false;
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(canvasWidth, canvasHeight);
}

function updateCanvasSize() {
  const container = document.querySelector('main');
  if (container) {
    canvasWidth = container.offsetWidth;
  }
}
