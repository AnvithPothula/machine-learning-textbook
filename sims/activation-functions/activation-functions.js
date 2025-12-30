// Activation Function Comparison MicroSim
// Interactive comparison of sigmoid, tanh, ReLU, and Leaky ReLU

// Canvas dimensions
let canvasWidth = 900;
let drawHeight = 650;
let controlHeight = 100;
let canvasHeight = drawHeight + controlHeight;
let margin = 40;
let sliderLeftMargin = 150;
let defaultTextSize = 16;

// Parameters
let xSlider, alphaSlider;
let derivCheckbox, saturationCheckbox;
let xValue = 0;
let leakyAlpha = 0.01;
let showDerivatives = false;
let showSaturation = true;

// Activation functions
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x) {
  const s = sigmoid(x);
  return s * (1 - s);
}

function tanhFunc(x) {
  return Math.tanh(x);
}

function tanhDerivative(x) {
  const t = Math.tanh(x);
  return 1 - t * t;
}

function relu(x) {
  return Math.max(0, x);
}

function reluDerivative(x) {
  return x > 0 ? 1 : 0;
}

function leakyRelu(x, alpha) {
  return x > 0 ? x : alpha * x;
}

function leakyReluDerivative(x, alpha) {
  return x > 0 ? 1 : alpha;
}

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.parent(document.querySelector('main'));

  // Create x value slider
  xSlider = createSlider(-10, 10, xValue, 0.1);
  xSlider.position(sliderLeftMargin, drawHeight + 15);
  xSlider.size(canvasWidth - sliderLeftMargin - margin - 350);

  // Create alpha slider
  alphaSlider = createSlider(0.01, 0.3, leakyAlpha, 0.01);
  alphaSlider.position(sliderLeftMargin, drawHeight + 45);
  alphaSlider.size(canvasWidth - sliderLeftMargin - margin - 350);

  // Create checkboxes
  derivCheckbox = createCheckbox('Show Derivatives', showDerivatives);
  derivCheckbox.position(canvasWidth - 330, drawHeight + 15);
  derivCheckbox.changed(() => {
    showDerivatives = derivCheckbox.checked();
  });

  saturationCheckbox = createCheckbox('Highlight Saturation', showSaturation);
  saturationCheckbox.position(canvasWidth - 330, drawHeight + 45);
  saturationCheckbox.changed(() => {
    showSaturation = saturationCheckbox.checked();
  });

  describe('Interactive visualization comparing sigmoid, tanh, ReLU, and Leaky ReLU activation functions', LABEL);
}

function draw() {
  updateCanvasSize();
  xValue = xSlider.value();
  leakyAlpha = alphaSlider.value();

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
  text('Activation Function Comparison', canvasWidth / 2, 10);

  // Reset text settings
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);

  // Draw four activation functions in a 2x2 grid
  let plotWidth = (canvasWidth - 3 * margin) / 2;
  let plotHeight = (drawHeight - 120) / 2;

  drawSigmoidPlot(margin, 50, plotWidth, plotHeight);
  drawTanhPlot(margin * 2 + plotWidth, 50, plotWidth, plotHeight);
  drawReLUPlot(margin, 70 + plotHeight, plotWidth, plotHeight);
  drawLeakyReLUPlot(margin * 2 + plotWidth, 70 + plotHeight, plotWidth, plotHeight);

  // Control labels
  fill('black');
  noStroke();
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('x value: ' + xValue.toFixed(2), 10, drawHeight + 20);
  text('Leaky ReLU α: ' + leakyAlpha.toFixed(2), 10, drawHeight + 50);

  // Draw instruction
  fill(100);
  textSize(13);
  text('Adjust x to see output values and derivatives', 10, drawHeight + 80);
}

function drawSigmoidPlot(x, y, w, h) {
  drawFunctionPlot(x, y, w, h, 'Sigmoid σ(x)', sigmoid, sigmoidDerivative,
                   color(156, 39, 176), -10, 10, 0, 1);
}

function drawTanhPlot(x, y, w, h) {
  drawFunctionPlot(x, y, w, h, 'Tanh', tanhFunc, tanhDerivative,
                   color(33, 150, 243), -10, 10, -1, 1);
}

function drawReLUPlot(x, y, w, h) {
  drawFunctionPlot(x, y, w, h, 'ReLU', relu, reluDerivative,
                   color(76, 175, 80), -10, 10, -2, 10);
}

function drawLeakyReLUPlot(x, y, w, h) {
  drawFunctionPlot(x, y, w, h, 'Leaky ReLU (α=' + leakyAlpha.toFixed(2) + ')',
                   (x) => leakyRelu(x, leakyAlpha),
                   (x) => leakyReluDerivative(x, leakyAlpha),
                   color(255, 152, 0), -10, 10, -2, 10);
}

function drawFunctionPlot(x, y, w, h, title, func, derivFunc, col, xMin, xMax, yMin, yMax) {
  push();
  translate(x, y);

  // Background
  fill(255);
  stroke(200);
  strokeWeight(1);
  rect(0, 0, w, h);

  // Title
  fill(col);
  noStroke();
  textSize(15);
  textAlign(CENTER, TOP);
  text(title, w / 2, 5);

  // Plot area
  let plotX = 40;
  let plotY = 35;
  let plotW = w - 60;
  let plotH = h - 80;

  // Axes
  stroke(100);
  strokeWeight(1);
  let zeroY = map(0, yMax, yMin, plotY, plotY + plotH);
  let zeroX = map(0, xMin, xMax, plotX, plotX + plotW);

  line(plotX, zeroY, plotX + plotW, zeroY); // X-axis
  line(zeroX, plotY, zeroX, plotY + plotH); // Y-axis

  // Draw function curve
  stroke(col);
  strokeWeight(2);
  noFill();
  beginShape();
  for (let px = 0; px <= plotW; px += 2) {
    let xVal = map(px, 0, plotW, xMin, xMax);
    let yVal = func(xVal);
    let py = map(yVal, yMax, yMin, plotY, plotY + plotH);
    py = constrain(py, plotY, plotY + plotH);
    vertex(plotX + px, py);
  }
  endShape();

  // Draw derivative if enabled
  if (showDerivatives) {
    stroke(col);
    strokeWeight(1);
    drawingContext.setLineDash([3, 3]);
    noFill();
    beginShape();
    for (let px = 0; px <= plotW; px += 2) {
      let xVal = map(px, 0, plotW, xMin, xMax);
      let yVal = derivFunc(xVal);
      let py = map(yVal, yMax, yMin, plotY, plotY + plotH);
      py = constrain(py, plotY, plotY + plotH);
      vertex(plotX + px, py);
    }
    endShape();
    drawingContext.setLineDash([]);
  }

  // Mark current x value
  let currentX = map(xValue, xMin, xMax, plotX, plotX + plotW);
  let currentY = map(func(xValue), yMax, yMin, plotY, plotY + plotH);
  currentY = constrain(currentY, plotY, plotY + plotH);

  // Draw vertical line at current x
  stroke(200);
  strokeWeight(1);
  drawingContext.setLineDash([2, 2]);
  line(currentX, plotY, currentX, plotY + plotH);
  drawingContext.setLineDash([]);

  // Draw point
  fill(col);
  stroke(255);
  strokeWeight(2);
  circle(currentX, currentY, 12);

  // Display values
  fill(0);
  noStroke();
  textSize(11);
  textAlign(LEFT, TOP);
  let output = func(xValue);
  let deriv = derivFunc(xValue);
  text(`f(${xValue.toFixed(1)}) = ${output.toFixed(3)}`, 5, h - 35);
  if (showDerivatives) {
    text(`f'(${xValue.toFixed(1)}) = ${deriv.toFixed(3)}`, 5, h - 20);
  }

  // Saturation warning for sigmoid/tanh
  if (showSaturation && (title.includes('Sigmoid') || title.includes('Tanh'))) {
    if (Math.abs(xValue) > 3) {
      fill(244, 67, 54, 100);
      noStroke();
      textSize(10);
      textAlign(CENTER, BOTTOM);
      text('⚠ Saturated!', w / 2, h - 5);
    }
  }

  pop();
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(canvasWidth, canvasHeight);
}

function updateCanvasSize() {
  const container = document.querySelector('main');
  if (container) {
    canvasWidth = container.offsetWidth;
    if (typeof xSlider !== 'undefined') {
      xSlider.size(canvasWidth - sliderLeftMargin - margin - 350);
      alphaSlider.size(canvasWidth - sliderLeftMargin - margin - 350);
      derivCheckbox.position(canvasWidth - 330, drawHeight + 15);
      saturationCheckbox.position(canvasWidth - 330, drawHeight + 45);
    }
  }
}
