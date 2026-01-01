// Convolution Operation MicroSim
// Demonstrates how convolution filters slide over images to detect features

// Canvas dimensions
let canvasWidth = 900;
let drawHeight = 650;
let controlHeight = 120;
let canvasHeight = drawHeight + controlHeight;
let margin = 20;
let sliderLeftMargin = 200;
let defaultTextSize = 16;

// Input image (5x5 grid)
let inputImage = [
  [0, 0, 1, 1, 1],
  [0, 0, 1, 1, 1],
  [1, 1, 1, 0, 0],
  [1, 1, 1, 0, 0],
  [1, 1, 1, 0, 0]
];

// Available filters
let filters = {
  'Vertical Edge': [
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
  ],
  'Horizontal Edge': [
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
  ],
  'Blur': [
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
  ],
  'Sharpen': [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
  ],
  'Identity': [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
  ]
};

// Parameters
let selectedFilter = 'Vertical Edge';
let animationSpeed = 1.0;
let showSteps = true;
let currentRow = 0;
let currentCol = 0;
let animationProgress = 0;
let isAnimating = false;

// Controls
let filterSelector, speedSlider, stepsCheckbox;
let playBtn, resetBtn;

// Output feature map
let featureMap = [];

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.parent(document.querySelector('main'));

  // Filter selector
  filterSelector = createSelect();
  for (let filterName in filters) {
    filterSelector.option(filterName);
  }
  filterSelector.selected('Vertical Edge');
  filterSelector.position(sliderLeftMargin, drawHeight + 15);
  filterSelector.changed(() => {
    selectedFilter = filterSelector.value();
    computeFeatureMap();
  });

  // Speed slider
  speedSlider = createSlider(0.1, 3.0, 1.0, 0.1);
  speedSlider.position(sliderLeftMargin, drawHeight + 50);
  speedSlider.size(canvasWidth - sliderLeftMargin - margin);

  // Show steps checkbox
  stepsCheckbox = createCheckbox('Show Steps', true);
  stepsCheckbox.position(sliderLeftMargin, drawHeight + 85);
  stepsCheckbox.changed(() => {
    showSteps = stepsCheckbox.checked();
  });

  // Play/Pause button
  playBtn = createButton('Play');
  playBtn.position(canvasWidth - 280, drawHeight + 13);
  playBtn.mousePressed(toggleAnimation);
  playBtn.size(80, 25);

  // Reset button
  resetBtn = createButton('Reset');
  resetBtn.position(canvasWidth - 190, drawHeight + 13);
  resetBtn.mousePressed(resetAnimation);
  resetBtn.size(80, 25);

  computeFeatureMap();

  describe('Interactive visualization of convolution operation showing how filters slide over images', LABEL);
}

function computeFeatureMap() {
  featureMap = [];
  let filter = filters[selectedFilter];
  let filterSize = 3;

  // Compute convolution for all valid positions
  for (let i = 0; i <= inputImage.length - filterSize; i++) {
    let row = [];
    for (let j = 0; j <= inputImage[0].length - filterSize; j++) {
      let sum = 0;
      for (let fi = 0; fi < filterSize; fi++) {
        for (let fj = 0; fj < filterSize; fj++) {
          sum += inputImage[i + fi][j + fj] * filter[fi][fj];
        }
      }
      row.push(sum);
    }
    featureMap.push(row);
  }
}

function toggleAnimation() {
  isAnimating = !isAnimating;
  playBtn.html(isAnimating ? 'Pause' : 'Play');
}

function resetAnimation() {
  isAnimating = false;
  currentRow = 0;
  currentCol = 0;
  animationProgress = 0;
  playBtn.html('Play');
}

function draw() {
  updateCanvasSize();

  // Update animation
  if (isAnimating) {
    animationProgress += speedSlider.value() * 0.02;
    if (animationProgress >= 1.0) {
      animationProgress = 0;
      currentCol++;
      if (currentCol >= featureMap[0].length) {
        currentCol = 0;
        currentRow++;
        if (currentRow >= featureMap.length) {
          currentRow = 0;
        }
      }
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
  text('Convolution Operation', canvasWidth / 2, 10);

  // Reset text settings
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);

  // Calculate cell sizes
  let cellSize = 50;
  let spacing = 80;

  // Draw input image
  let inputX = 50;
  let inputY = 80;
  drawGrid(inputX, inputY, inputImage, cellSize, 'Input Image (5×5)', color(100, 150, 255));

  // Draw filter
  let filterX = inputX + (inputImage[0].length * cellSize) + spacing;
  let filterY = inputY + 50;
  drawFilter(filterX, filterY, filters[selectedFilter], cellSize, selectedFilter + ' Filter (3×3)');

  // Draw current convolution position
  if (showSteps) {
    push();
    noFill();
    stroke(255, 100, 0);
    strokeWeight(4);
    rect(inputX + currentCol * cellSize, inputY + 30 + currentRow * cellSize, cellSize * 3, cellSize * 3);
    pop();

    // Draw computation steps
    drawComputationSteps(filterX, filterY + 200, currentRow, currentCol, cellSize);
  }

  // Draw feature map
  let featureX = inputX;
  let featureY = inputY + (inputImage.length * cellSize) + 100;
  drawGrid(featureX, featureY, featureMap, cellSize, 'Output Feature Map (3×3)', color(100, 255, 150));

  // Highlight current output position
  if (showSteps && featureMap.length > 0) {
    push();
    noFill();
    stroke(255, 100, 0);
    strokeWeight(3);
    rect(featureX + currentCol * cellSize, featureY + 30 + currentRow * cellSize, cellSize, cellSize);
    pop();
  }

  // Control labels
  fill('black');
  noStroke();
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Filter Type:', 10, drawHeight + 20);
  text('Animation Speed: ' + speedSlider.value().toFixed(1), 10, drawHeight + 55);
  text('Options:', 10, drawHeight + 90);
}

function drawGrid(x, y, grid, cellSize, title, highlightColor) {
  push();

  // Title
  fill('black');
  textAlign(LEFT, TOP);
  textSize(14);
  textStyle(BOLD);
  text(title, x, y);

  textStyle(NORMAL);

  // Grid
  for (let i = 0; i < grid.length; i++) {
    for (let j = 0; j < grid[i].length; j++) {
      let value = grid[i][j];

      // Cell background
      if (value >= 0) {
        fill(255 - value * 50, 255 - value * 50, 255);
      } else {
        fill(255, 255 - Math.abs(value) * 50, 255 - Math.abs(value) * 50);
      }
      stroke(100);
      strokeWeight(1);
      rect(x + j * cellSize, y + 30 + i * cellSize, cellSize, cellSize);

      // Cell value
      fill('black');
      noStroke();
      textAlign(CENTER, CENTER);
      textSize(12);
      text(value.toFixed(1), x + j * cellSize + cellSize/2, y + 30 + i * cellSize + cellSize/2);
    }
  }

  pop();
}

function drawFilter(x, y, filter, cellSize, title) {
  push();

  // Title
  fill('black');
  textAlign(LEFT, TOP);
  textSize(14);
  textStyle(BOLD);
  text(title, x, y);

  textStyle(NORMAL);

  // Filter grid
  for (let i = 0; i < filter.length; i++) {
    for (let j = 0; j < filter[i].length; j++) {
      let value = filter[i][j];

      // Cell background
      if (value > 0) {
        fill(200, 255, 200);
      } else if (value < 0) {
        fill(255, 200, 200);
      } else {
        fill(240);
      }
      stroke(100);
      strokeWeight(1);
      rect(x + j * cellSize, y + 30 + i * cellSize, cellSize, cellSize);

      // Cell value
      fill('black');
      noStroke();
      textAlign(CENTER, CENTER);
      textSize(12);
      text(value.toFixed(2), x + j * cellSize + cellSize/2, y + 30 + i * cellSize + cellSize/2);
    }
  }

  pop();
}

function drawComputationSteps(x, y, row, col, cellSize) {
  push();

  fill('black');
  textAlign(LEFT, TOP);
  textSize(13);
  textStyle(BOLD);
  text('Computation at Position (' + row + ', ' + col + '):', x, y);

  textStyle(NORMAL);
  textSize(11);

  let filter = filters[selectedFilter];
  let sum = 0;
  let stepY = y + 25;

  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let inputVal = inputImage[row + i][col + j];
      let filterVal = filter[i][j];
      let product = inputVal * filterVal;
      sum += product;

      text(inputVal.toFixed(1) + ' × ' + filterVal.toFixed(2) + ' = ' + product.toFixed(2), x + 10, stepY);
      stepY += 15;
    }
  }

  strokeWeight(1);
  stroke(100);
  line(x, stepY, x + 150, stepY);
  stepY += 5;

  textStyle(BOLD);
  fill(255, 100, 0);
  text('Sum = ' + sum.toFixed(2), x + 10, stepY);

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
    if (typeof filterSelector !== 'undefined') {
      filterSelector.position(sliderLeftMargin, drawHeight + 15);
      speedSlider.position(sliderLeftMargin, drawHeight + 50);
      speedSlider.size(canvasWidth - sliderLeftMargin - margin);
      stepsCheckbox.position(sliderLeftMargin, drawHeight + 85);
      playBtn.position(canvasWidth - 280, drawHeight + 13);
      resetBtn.position(canvasWidth - 190, drawHeight + 13);
    }
  }
}
