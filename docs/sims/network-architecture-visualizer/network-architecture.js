// Neural Network Architecture Visualizer
// Shows how different neural network architectures compare

// Canvas dimensions
let canvasWidth = 900;
let drawHeight = 650;
let controlHeight = 100;
let canvasHeight = drawHeight + controlHeight;
let margin = 40;
let sliderLeftMargin = 150;
let defaultTextSize = 16;

// Network structure
let layers = [4, 8, 6, 3]; // neurons per layer
let neuronPositions = [];
let connections = [];
let activations = [];
let animate = false;
let animationStep = 0;

// Controls
let layer1Slider, layer2Slider;
let presetSelector, animateBtn, resetBtn;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.parent(document.querySelector('main'));

  // Hidden layer 1 slider
  layer1Slider = createSlider(2, 16, 8, 1);
  layer1Slider.position(sliderLeftMargin, drawHeight + 15);
  layer1Slider.size(200);

  // Hidden layer 2 slider
  layer2Slider = createSlider(2, 16, 6, 1);
  layer2Slider.position(sliderLeftMargin, drawHeight + 50);
  layer2Slider.size(200);

  // Preset selector
  presetSelector = createSelect();
  presetSelector.option('Custom');
  presetSelector.option('Shallow (4-8-3)');
  presetSelector.option('Deep (4-6-6-6-3)');
  presetSelector.option('Wide (4-16-3)');
  presetSelector.selected('Custom');
  presetSelector.position(canvasWidth - 380, drawHeight + 15);
  presetSelector.changed(loadPreset);

  // Animate button
  animateBtn = createButton('Animate');
  animateBtn.position(canvasWidth - 220, drawHeight + 13);
  animateBtn.mousePressed(startAnimation);
  animateBtn.size(100, 25);

  // Reset button
  resetBtn = createButton('Reset');
  resetBtn.position(canvasWidth - 110, drawHeight + 13);
  resetBtn.mousePressed(resetNetwork);
  resetBtn.size(90, 25);

  initializeNetwork();

  describe('Interactive visualization of neural network architecture with adjustable layers and neurons', LABEL);
}

function loadPreset() {
  const preset = presetSelector.value();
  if (preset === 'Shallow (4-8-3)') {
    layers = [4, 8, 3];
    layer1Slider.value(8);
    layer2Slider.value(3);
  } else if (preset === 'Deep (4-6-6-6-3)') {
    layers = [4, 6, 6, 6, 3];
    layer1Slider.value(6);
    layer2Slider.value(6);
  } else if (preset === 'Wide (4-16-3)') {
    layers = [4, 16, 3];
    layer1Slider.value(16);
    layer2Slider.value(3);
  }
  initializeNetwork();
}

function startAnimation() {
  animate = true;
  animationStep = 0;
  randomizeInputs();
}

function resetNetwork() {
  layers = [4, 8, 6, 3];
  layer1Slider.value(8);
  layer2Slider.value(6);
  presetSelector.selected('Custom');
  initializeNetwork();
}

function initializeNetwork() {
  neuronPositions = [];
  connections = [];
  activations = [];

  const drawWidth = canvasWidth - 220;
  const netMargin = 50;
  const layerSpacing = (drawWidth - 2 * netMargin) / (layers.length - 1);

  // Calculate neuron positions
  for (let l = 0; l < layers.length; l++) {
    const layerSize = layers[l];
    const x = netMargin + l * layerSpacing;
    const verticalSpacing = (drawHeight - 200) / (layerSize + 1);

    neuronPositions[l] = [];
    activations[l] = [];

    for (let n = 0; n < layerSize; n++) {
      const y = 100 + (n + 1) * verticalSpacing;
      neuronPositions[l].push({ x, y });
      activations[l].push(l === 0 ? random(0.2, 0.8) : 0);
    }
  }

  // Create connections
  connections = [];
  for (let l = 0; l < layers.length - 1; l++) {
    for (let i = 0; i < layers[l]; i++) {
      for (let j = 0; j < layers[l + 1]; j++) {
        connections.push({
          from: neuronPositions[l][i],
          to: neuronPositions[l + 1][j],
          weight: random(-1, 1),
          fromLayer: l,
          toLayer: l + 1,
          fromIdx: i,
          toIdx: j
        });
      }
    }
  }

  animate = false;
}

function randomizeInputs() {
  for (let i = 0; i < activations[0].length; i++) {
    activations[0][i] = random(0.2, 0.8);
  }
}

function draw() {
  updateCanvasSize();

  // Update layers from sliders
  const newLayer1 = layer1Slider.value();
  const newLayer2 = layer2Slider.value();

  if (newLayer1 !== layers[1] || newLayer2 !== layers[2]) {
    layers = [4, newLayer1, newLayer2, 3];
    presetSelector.selected('Custom');
    initializeNetwork();
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
  text('Neural Network Architecture Visualizer', canvasWidth / 2, 10);

  // Reset text settings
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);

  // Handle animation
  if (animate) {
    updateAnimation();
  }

  // Draw network
  drawConnections();
  drawNeurons();
  drawInfoPanel();

  // Control labels
  fill('black');
  noStroke();
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Hidden Layer 1: ' + layers[1], 10, drawHeight + 20);
  text('Hidden Layer 2: ' + layers[2], 10, drawHeight + 55);

  // Draw instruction
  fill(100);
  textSize(13);
  text('Adjust layers to see different architectures', 10, drawHeight + 85);
}

function drawConnections() {
  const showAll = layers.reduce((sum, l) => sum + l, 0) < 30;

  for (let conn of connections) {
    // Only show subset for large networks
    if (!showAll && abs(conn.weight) < 0.5) continue;

    const alpha = animate && conn.fromLayer < Math.floor(animationStep) ? 150 : 40;
    const weight = conn.weight;

    if (weight > 0) {
      stroke(76, 175, 80, alpha);
    } else {
      stroke(244, 67, 54, alpha);
    }

    strokeWeight(map(abs(weight), 0, 1, 0.5, 2));
    line(conn.from.x, conn.from.y, conn.to.x, conn.to.y);
  }
}

function drawNeurons() {
  const layerColors = [
    color(33, 150, 243),   // Input - blue
    color(76, 175, 80),    // Hidden - green
    color(76, 175, 80),    // Hidden - green
    color(255, 152, 0)     // Output - orange
  ];

  for (let l = 0; l < neuronPositions.length; l++) {
    const col = layerColors[min(l, layerColors.length - 1)];

    for (let n = 0; n < neuronPositions[l].length; n++) {
      const pos = neuronPositions[l][n];
      const activation = activations[l][n];

      // Neuron circle
      if (animate && l <= Math.floor(animationStep)) {
        fill(red(col), green(col), blue(col), map(activation, 0, 1, 50, 255));
      } else {
        fill(red(col), green(col), blue(col), 180);
      }
      stroke(100);
      strokeWeight(2);
      circle(pos.x, pos.y, 20);

      // Activation value (if animated)
      if (animate && l <= Math.floor(animationStep)) {
        fill(0);
        noStroke();
        textAlign(CENTER, CENTER);
        textSize(8);
        text(activation.toFixed(2), pos.x, pos.y);
      }
    }
  }

  // Layer labels
  for (let l = 0; l < neuronPositions.length; l++) {
    if (neuronPositions[l].length > 0) {
      fill(0);
      noStroke();
      textAlign(CENTER, TOP);
      textSize(12);
      textStyle(BOLD);
      const name = l === 0 ? 'Input' :
                   l === layers.length - 1 ? 'Output' :
                   `Hidden ${l}`;
      text(name, neuronPositions[l][0].x, 50);

      textStyle(NORMAL);
      textSize(10);
      text(`(${layers[l]})`, neuronPositions[l][0].x, 66);
    }
  }
}

function drawInfoPanel() {
  const panelX = canvasWidth - 200;
  const panelY = 100;
  const panelW = 180;
  const panelH = drawHeight - 220;

  // Background
  fill(255, 255, 255, 240);
  stroke(200);
  strokeWeight(1);
  rect(panelX, panelY, panelW, panelH, 10);

  // Title
  fill(0);
  noStroke();
  textAlign(CENTER, TOP);
  textSize(14);
  textStyle(BOLD);
  text('Network Info', panelX + panelW / 2, panelY + 15);

  // Architecture
  textAlign(LEFT, TOP);
  textSize(11);
  textStyle(BOLD);
  text('Architecture:', panelX + 15, panelY + 50);

  textStyle(NORMAL);
  textSize(10);
  let archText = layers.join(' → ');
  text(archText, panelX + 15, panelY + 70, panelW - 30);

  // Parameter count
  let params = 0;
  for (let l = 0; l < layers.length - 1; l++) {
    params += (layers[l] + 1) * layers[l + 1]; // +1 for bias
  }

  textStyle(BOLD);
  textSize(11);
  text('Parameters:', panelX + 15, panelY + 110);
  textStyle(NORMAL);
  textSize(10);
  text(params.toLocaleString(), panelX + 15, panelY + 130);

  // Depth
  textStyle(BOLD);
  textSize(11);
  text('Depth:', panelX + 15, panelY + 160);
  textStyle(NORMAL);
  textSize(10);
  text(`${layers.length} layers`, panelX + 15, panelY + 180);

  // Width
  textStyle(BOLD);
  textSize(11);
  text('Max Width:', panelX + 15, panelY + 210);
  textStyle(NORMAL);
  textSize(10);
  text(`${Math.max(...layers)} neurons`, panelX + 15, panelY + 230);

  // Connection count
  textStyle(BOLD);
  textSize(11);
  text('Connections:', panelX + 15, panelY + 260);
  textStyle(NORMAL);
  textSize(10);
  text(connections.length.toLocaleString(), panelX + 15, panelY + 280);

  // Legend
  textStyle(BOLD);
  textSize(11);
  text('Layer Types:', panelX + 15, panelY + 320);

  textStyle(NORMAL);
  textSize(10);
  fill(33, 150, 243);
  circle(panelX + 25, panelY + 345, 12);
  fill(0);
  text('Input', panelX + 35, panelY + 340);

  fill(76, 175, 80);
  circle(panelX + 25, panelY + 365, 12);
  fill(0);
  text('Hidden', panelX + 35, panelY + 360);

  fill(255, 152, 0);
  circle(panelX + 25, panelY + 385, 12);
  fill(0);
  text('Output', panelX + 35, panelY + 380);
}

function updateAnimation() {
  animationStep += 0.02;

  if (animationStep >= layers.length) {
    animate = false;
    animationStep = 0;
    return;
  }

  // Propagate activations
  const currentLayer = Math.floor(animationStep);
  if (currentLayer > 0 && currentLayer < layers.length) {
    for (let j = 0; j < layers[currentLayer]; j++) {
      let sum = 0;
      for (let i = 0; i < layers[currentLayer - 1]; i++) {
        const conn = connections.find(c =>
          c.fromLayer === currentLayer - 1 &&
          c.toLayer === currentLayer &&
          c.fromIdx === i &&
          c.toIdx === j
        );
        if (conn) {
          sum += activations[currentLayer - 1][i] * conn.weight;
        }
      }
      // ReLU activation
      activations[currentLayer][j] = max(0, sum / layers[currentLayer - 1] + 0.1);
    }
  }
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(canvasWidth, canvasHeight);
}

function updateCanvasSize() {
  const container = document.querySelector('main');
  if (container) {
    canvasWidth = container.offsetWidth;
    if (typeof layer1Slider !== 'undefined') {
      layer1Slider.position(sliderLeftMargin, drawHeight + 15);
      layer2Slider.position(sliderLeftMargin, drawHeight + 50);
      presetSelector.position(canvasWidth - 380, drawHeight + 15);
      animateBtn.position(canvasWidth - 220, drawHeight + 13);
      resetBtn.position(canvasWidth - 110, drawHeight + 13);
      initializeNetwork();
    }
  }
}
