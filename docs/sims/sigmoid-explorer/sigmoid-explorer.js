// Sigmoid Function Explorer MicroSim
// Shows how sigmoid transforms linear z = mx + b into probabilities

// Canvas dimensions
let canvasWidth = 800;
let drawHeight = 600;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let margin = 40;
let sliderLeftMargin = 200;
let defaultTextSize = 16;

// Parameters
let slopeSlider, interceptSlider;
let slope = 1;
let intercept = 0;
let resetButton;

// Sample data points for visualization
let samplePoints = [];

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.parent(document.querySelector('main'));

  // Generate sample data points
  generateSamplePoints();

  // Create slope slider
  slopeSlider = createSlider(-5, 5, slope, 0.1);
  slopeSlider.position(sliderLeftMargin, drawHeight + 15);
  slopeSlider.size(canvasWidth - sliderLeftMargin - margin);

  // Create intercept slider
  interceptSlider = createSlider(-5, 5, intercept, 0.1);
  interceptSlider.position(sliderLeftMargin, drawHeight + 45);
  interceptSlider.size(canvasWidth - sliderLeftMargin - margin);

  // Create reset button
  resetButton = createButton('Reset');
  resetButton.position(10, drawHeight + 15);
  resetButton.mousePressed(resetParameters);
  resetButton.size(80, 25);

  describe('Interactive visualization showing how sigmoid function transforms linear model outputs into probabilities', LABEL);
}

function draw() {
  updateCanvasSize();
  slope = slopeSlider.value();
  intercept = interceptSlider.value();

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
  textSize(22);
  textAlign(CENTER, TOP);
  text('Sigmoid Function Explorer', canvasWidth / 2, 10);

  // Reset text settings
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);

  // Draw the two side-by-side plots
  let plotWidth = (canvasWidth - 3 * margin) / 2;
  let plotHeight = drawHeight - 120;
  let plotY = 80;

  // Left plot: Linear function z = mx + b
  drawLinearPlot(margin, plotY, plotWidth, plotHeight);

  // Right plot: Sigmoid function σ(z)
  drawSigmoidPlot(margin * 2 + plotWidth, plotY, plotWidth, plotHeight);

  // Draw arrow connecting plots
  drawConnectionArrow(margin + plotWidth, plotY + plotHeight/2, margin * 2 + plotWidth, plotY + plotHeight/2);

  // Draw control labels
  fill('black');
  noStroke();
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Slope (m): ' + slope.toFixed(1), 110, drawHeight + 20);
  text('Intercept (b): ' + intercept.toFixed(1), 110, drawHeight + 50);
}

function drawLinearPlot(x, y, w, h) {
  push();
  translate(x, y);

  // Title
  fill(33, 150, 243);
  noStroke();
  textSize(18);
  textAlign(CENTER, TOP);
  text('Linear: z = mx + b', w / 2, -30);

  // Plot area
  fill(255);
  stroke(200);
  strokeWeight(1);
  rect(0, 0, w, h);

  // Axes
  stroke(100);
  strokeWeight(1);
  let centerY = h / 2;
  let centerX = w / 2;
  line(0, centerY, w, centerY); // X-axis
  line(centerX, 0, centerX, h); // Y-axis

  // Axis labels
  fill(0);
  noStroke();
  textSize(14);
  textAlign(CENTER, TOP);
  text('x', w - 10, centerY + 5);
  textAlign(RIGHT, CENTER);
  text('z', centerX - 5, 10);

  // Draw linear function
  stroke(33, 150, 243);
  strokeWeight(3);
  noFill();
  beginShape();
  for (let px = 0; px <= w; px += 2) {
    let xVal = map(px, 0, w, -5, 5);
    let zVal = slope * xVal + intercept;
    let py = map(zVal, -10, 10, h, 0);
    py = constrain(py, 0, h);
    vertex(px, py);
  }
  endShape();

  // Draw sample points
  for (let pt of samplePoints) {
    let px = map(pt.x, -5, 5, 0, w);
    let z = slope * pt.x + intercept;
    let py = map(z, -10, 10, h, 0);

    if (py >= 0 && py <= h) {
      fill(pt.label === 1 ? color(76, 175, 80) : color(244, 67, 54));
      noStroke();
      circle(px, py, 10);
    }
  }

  pop();
}

function drawSigmoidPlot(x, y, w, h) {
  push();
  translate(x, y);

  // Title
  fill(156, 39, 176);
  noStroke();
  textSize(18);
  textAlign(CENTER, TOP);
  text('Sigmoid: σ(z) = 1/(1+e⁻ᶻ)', w / 2, -30);

  // Plot area
  fill(255);
  stroke(200);
  strokeWeight(1);
  rect(0, 0, w, h);

  // Axes
  stroke(100);
  strokeWeight(1);
  let centerX = w / 2;
  line(0, h, w, h); // Bottom (probability = 0)
  line(0, 0, w, 0); // Top (probability = 1)
  line(centerX, 0, centerX, h); // Z-axis

  // Probability labels
  fill(0);
  noStroke();
  textSize(14);
  textAlign(RIGHT, CENTER);
  text('P=1', w - 5, 15);
  text('P=0.5', w - 5, h / 2);
  text('P=0', w - 5, h - 15);

  // Threshold lines
  stroke(200);
  strokeWeight(1);
  drawingContext.setLineDash([3, 3]);
  line(0, h / 2, w, h / 2); // P = 0.5
  drawingContext.setLineDash([]);

  textAlign(CENTER, TOP);
  text('z', centerX, h + 5);

  // Draw sigmoid curve
  stroke(156, 39, 176);
  strokeWeight(3);
  noFill();
  beginShape();
  for (let px = 0; px <= w; px += 2) {
    let z = map(px, 0, w, -10, 10);
    let sigma = 1 / (1 + Math.exp(-z));
    let py = map(sigma, 0, 1, h, 0);
    vertex(px, py);
  }
  endShape();

  // Draw transformed sample points
  for (let pt of samplePoints) {
    let z = slope * pt.x + intercept;
    let sigma = 1 / (1 + Math.exp(-z));
    let px = map(z, -10, 10, 0, w);
    let py = map(sigma, 0, 1, h, 0);

    if (px >= 0 && px <= w) {
      fill(pt.label === 1 ? color(76, 175, 80) : color(244, 67, 54));
      noStroke();
      circle(px, py, 10);

      // Draw probability text
      fill(0);
      textSize(11);
      textAlign(CENTER, BOTTOM);
      text(sigma.toFixed(2), px, py - 8);
    }
  }

  pop();
}

function drawConnectionArrow(x1, y1, x2, y2) {
  push();
  stroke(100);
  strokeWeight(2);
  drawingContext.setLineDash([5, 5]);
  line(x1, y1, x2, y2);
  drawingContext.setLineDash([]);

  // Arrow head
  let arrowSize = 10;
  fill(100);
  noStroke();
  triangle(x2, y2, x2 - arrowSize, y2 - arrowSize/2, x2 - arrowSize, y2 + arrowSize/2);

  // Label
  fill(100);
  textSize(14);
  textAlign(CENTER, BOTTOM);
  text('Transform', (x1 + x2) / 2, y1 - 10);
  pop();
}

function generateSamplePoints() {
  samplePoints = [];
  randomSeed(42); // Consistent points
  for (let i = 0; i < 12; i++) {
    samplePoints.push({
      x: random(-4, 4),
      label: random() > 0.5 ? 1 : 0
    });
  }
}

function resetParameters() {
  slope = 1;
  intercept = 0;
  slopeSlider.value(slope);
  interceptSlider.value(intercept);
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(canvasWidth, canvasHeight);
}

function updateCanvasSize() {
  const container = document.querySelector('main');
  if (container) {
    canvasWidth = container.offsetWidth;
    if (typeof slopeSlider !== 'undefined') {
      slopeSlider.size(canvasWidth - sliderLeftMargin - margin);
      interceptSlider.size(canvasWidth - sliderLeftMargin - margin);
    }
  }
}
