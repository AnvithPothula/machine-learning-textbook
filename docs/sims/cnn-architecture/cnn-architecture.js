// CNN Architecture Visualizer
// Shows the structure and data flow through a convolutional neural network

// Canvas dimensions
let canvasWidth = 900;
let drawHeight = 650;
let controlHeight = 100;
let canvasHeight = drawHeight + controlHeight;
let margin = 20;
let sliderLeftMargin = 180;
let defaultTextSize = 16;

// CNN Architecture layers
let architecture = {
  'Simple CNN': [
    {type: 'input', size: [32, 32, 3], name: 'Input'},
    {type: 'conv', filters: 16, kernel: 3, size: [30, 30, 16], name: 'Conv1'},
    {type: 'pool', size: [15, 15, 16], name: 'MaxPool1'},
    {type: 'conv', filters: 32, kernel: 3, size: [13, 13, 32], name: 'Conv2'},
    {type: 'pool', size: [6, 6, 32], name: 'MaxPool2'},
    {type: 'fc', neurons: 128, name: 'FC1'},
    {type: 'output', neurons: 10, name: 'Output'}
  ],
  'VGG-like': [
    {type: 'input', size: [224, 224, 3], name: 'Input'},
    {type: 'conv', filters: 64, kernel: 3, size: [224, 224, 64], name: 'Conv1'},
    {type: 'conv', filters: 64, kernel: 3, size: [224, 224, 64], name: 'Conv2'},
    {type: 'pool', size: [112, 112, 64], name: 'Pool1'},
    {type: 'conv', filters: 128, kernel: 3, size: [112, 112, 128], name: 'Conv3'},
    {type: 'pool', size: [56, 56, 128], name: 'Pool2'},
    {type: 'fc', neurons: 4096, name: 'FC1'},
    {type: 'fc', neurons: 1000, name: 'Output'}
  ],
  'ResNet Block': [
    {type: 'input', size: [56, 56, 64], name: 'Input'},
    {type: 'conv', filters: 64, kernel: 1, size: [56, 56, 64], name: 'Conv1×1'},
    {type: 'conv', filters: 64, kernel: 3, size: [56, 56, 64], name: 'Conv3×3'},
    {type: 'conv', filters: 256, kernel: 1, size: [56, 56, 256], name: 'Conv1×1'},
    {type: 'add', size: [56, 56, 256], name: 'Add'},
    {type: 'output', size: [56, 56, 256], name: 'ReLU'}
  ]
};

// Current architecture
let currentArch = 'Simple CNN';
let layers = architecture[currentArch];

// Animation
let dataFlowProgress = 0;
let isAnimating = false;
let animationSpeed = 0.5;

// Controls
let archSelector, speedSlider;
let playBtn, resetBtn;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.parent(document.querySelector('main'));

  // Architecture selector
  archSelector = createSelect();
  for (let archName in architecture) {
    archSelector.option(archName);
  }
  archSelector.selected('Simple CNN');
  archSelector.position(sliderLeftMargin, drawHeight + 15);
  archSelector.changed(() => {
    currentArch = archSelector.value();
    layers = architecture[currentArch];
    resetAnimation();
  });

  // Speed slider
  speedSlider = createSlider(0.1, 2.0, 0.5, 0.1);
  speedSlider.position(sliderLeftMargin, drawHeight + 50);
  speedSlider.size(canvasWidth - sliderLeftMargin - margin);

  // Play/Pause button
  playBtn = createButton('Play');
  playBtn.position(canvasWidth - 220, drawHeight + 13);
  playBtn.mousePressed(toggleAnimation);
  playBtn.size(100, 25);

  // Reset button
  resetBtn = createButton('Reset');
  resetBtn.position(canvasWidth - 110, drawHeight + 13);
  resetBtn.mousePressed(resetAnimation);
  resetBtn.size(90, 25);

  describe('Interactive visualization of CNN architecture showing layers and data flow', LABEL);
}

function toggleAnimation() {
  isAnimating = !isAnimating;
  playBtn.html(isAnimating ? 'Pause' : 'Play');
}

function resetAnimation() {
  isAnimating = false;
  dataFlowProgress = 0;
  playBtn.html('Play');
}

function draw() {
  updateCanvasSize();

  // Update animation
  if (isAnimating) {
    dataFlowProgress += speedSlider.value() * 0.005;
    if (dataFlowProgress > 1.0) {
      dataFlowProgress = 0;
    }
  }

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
  text('CNN Architecture: ' + currentArch, canvasWidth / 2, 10);

  // Reset text settings
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);

  // Draw architecture
  drawArchitecture();

  // Draw legend
  drawLegend();

  // Control labels
  fill('black');
  noStroke();
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Architecture:', 10, drawHeight + 20);
  text('Flow Speed: ' + speedSlider.value().toFixed(1), 10, drawHeight + 55);
}

function drawArchitecture() {
  let availableWidth = canvasWidth - 2 * margin - 200;
  let layerSpacing = availableWidth / (layers.length + 1);
  let startX = margin + layerSpacing;
  let centerY = drawHeight / 2;

  // Draw connections between layers
  stroke(150);
  strokeWeight(2);
  for (let i = 0; i < layers.length - 1; i++) {
    let x1 = startX + i * layerSpacing;
    let x2 = startX + (i + 1) * layerSpacing;

    // Animated data flow
    if (isAnimating) {
      let flowPos = dataFlowProgress * (layers.length - 1);
      if (flowPos >= i && flowPos < i + 1) {
        stroke(76, 175, 80);
        strokeWeight(4);
        let t = flowPos - i;
        let flowX = lerp(x1, x2, t);
        circle(flowX, centerY, 15);
        strokeWeight(2);
      }
    }

    stroke(150);
    line(x1, centerY, x2, centerY);
  }

  // Draw layers
  for (let i = 0; i < layers.length; i++) {
    let layer = layers[i];
    let x = startX + i * layerSpacing;

    drawLayer(x, centerY, layer);
  }
}

function drawLayer(x, y, layer) {
  push();

  // Determine layer dimensions for visualization
  let layerWidth, layerHeight, layerDepth;

  if (layer.size) {
    // Scale down for visualization
    let maxDim = 120;
    let scale = maxDim / Math.max(layer.size[0], layer.size[1]);
    layerWidth = layer.size[1] * scale * 0.3;
    layerHeight = layer.size[0] * scale * 0.3;
    layerDepth = Math.min(layer.size[2] * 2, 40);
  } else if (layer.neurons) {
    layerWidth = 40;
    layerHeight = 80;
    layerDepth = 10;
  }

  // Layer color based on type
  let layerColor;
  switch(layer.type) {
    case 'input':
      layerColor = color(100, 150, 255);
      break;
    case 'conv':
      layerColor = color(76, 175, 80);
      break;
    case 'pool':
      layerColor = color(255, 152, 0);
      break;
    case 'fc':
      layerColor = color(156, 39, 176);
      break;
    case 'output':
      layerColor = color(244, 67, 54);
      break;
    case 'add':
      layerColor = color(0, 188, 212);
      break;
    default:
      layerColor = color(150);
  }

  // Draw 3D-ish layer representation
  fill(layerColor);
  stroke(100);
  strokeWeight(1);

  // Main rectangle
  rect(x - layerWidth/2, y - layerHeight/2, layerWidth, layerHeight);

  // Depth effect
  if (layerDepth > 5) {
    let offset = layerDepth;
    fill(red(layerColor) * 0.7, green(layerColor) * 0.7, blue(layerColor) * 0.7);
    quad(
      x - layerWidth/2, y - layerHeight/2,
      x - layerWidth/2 + offset, y - layerHeight/2 - offset,
      x + layerWidth/2 + offset, y - layerHeight/2 - offset,
      x + layerWidth/2, y - layerHeight/2
    );
    quad(
      x + layerWidth/2, y - layerHeight/2,
      x + layerWidth/2 + offset, y - layerHeight/2 - offset,
      x + layerWidth/2 + offset, y + layerHeight/2 - offset,
      x + layerWidth/2, y + layerHeight/2
    );
  }

  // Layer name
  fill('black');
  noStroke();
  textAlign(CENTER, TOP);
  textSize(11);
  textStyle(BOLD);
  text(layer.name, x, y + layerHeight/2 + 5);

  // Layer details
  textStyle(NORMAL);
  textSize(9);
  if (layer.size) {
    text(layer.size[0] + '×' + layer.size[1] + '×' + layer.size[2], x, y + layerHeight/2 + 20);
  } else if (layer.neurons) {
    text(layer.neurons + ' neurons', x, y + layerHeight/2 + 20);
  }

  if (layer.filters) {
    text(layer.filters + ' filters', x, y + layerHeight/2 + 32);
  }

  pop();
}

function drawLegend() {
  let legendX = canvasWidth - 180;
  let legendY = 100;
  let legendW = 160;
  let legendH = 280;

  // Background
  fill(255, 255, 255, 230);
  stroke(200);
  strokeWeight(1);
  rect(legendX, legendY, legendW, legendH, 10);

  // Title
  fill('black');
  noStroke();
  textAlign(CENTER, TOP);
  textSize(12);
  textStyle(BOLD);
  text('Layer Types', legendX + legendW/2, legendY + 10);

  textStyle(NORMAL);
  textAlign(LEFT, CENTER);
  textSize(10);

  let y = legendY + 35;
  let spacing = 35;

  // Input
  fill(100, 150, 255);
  rect(legendX + 15, y - 8, 20, 16);
  fill('black');
  text('Input Image', legendX + 45, y);
  y += spacing;

  // Conv
  fill(76, 175, 80);
  rect(legendX + 15, y - 8, 20, 16);
  fill('black');
  text('Convolution', legendX + 45, y);
  y += spacing;

  // Pool
  fill(255, 152, 0);
  rect(legendX + 15, y - 8, 20, 16);
  fill('black');
  text('Pooling', legendX + 45, y);
  y += spacing;

  // FC
  fill(156, 39, 176);
  rect(legendX + 15, y - 8, 20, 16);
  fill('black');
  text('Fully Connected', legendX + 45, y);
  y += spacing;

  // Output
  fill(244, 67, 54);
  rect(legendX + 15, y - 8, 20, 16);
  fill('black');
  text('Output', legendX + 45, y);
  y += spacing;

  // Add
  fill(0, 188, 212);
  rect(legendX + 15, y - 8, 20, 16);
  fill('black');
  text('Add (ResNet)', legendX + 45, y);
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(canvasWidth, canvasHeight);
}

function updateCanvasSize() {
  const container = document.querySelector('main');
  if (container) {
    canvasWidth = container.offsetWidth;
    if (typeof archSelector !== 'undefined') {
      archSelector.position(sliderLeftMargin, drawHeight + 15);
      speedSlider.position(sliderLeftMargin, drawHeight + 50);
      speedSlider.size(canvasWidth - sliderLeftMargin - margin);
      playBtn.position(canvasWidth - 220, drawHeight + 13);
      resetBtn.position(canvasWidth - 110, drawHeight + 13);
    }
  }
}
