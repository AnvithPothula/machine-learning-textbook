// K-Fold Cross-Validation Visualization MicroSim
// Shows how K-fold partitions data into training and validation sets

// Canvas dimensions
let canvasWidth = 800;
let drawHeight = 600;
let controlHeight = 110;
let canvasHeight = drawHeight + controlHeight;
let margin = 40;
let sliderLeftMargin = 180;
let defaultTextSize = 16;

// Parameters
let kSlider;
let numFolds = 5;
let currentFold = 1;
let nextButton, resetButton, runAllButton;
let totalSamples = 50;

// Colors
let trainColor;
let valColor;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.parent(document.querySelector('main'));

  trainColor = color(33, 150, 243);  // Blue
  valColor = color(255, 152, 0);     // Orange

  // Create k slider
  kSlider = createSlider(3, 10, numFolds, 1);
  kSlider.position(sliderLeftMargin, drawHeight + 15);
  kSlider.size(canvasWidth - sliderLeftMargin - margin - 270);

  // Create buttons
  nextButton = createButton('Next Fold');
  nextButton.position(canvasWidth - 260, drawHeight + 13);
  nextButton.mousePressed(nextFold);
  nextButton.size(80, 25);

  runAllButton = createButton('Run All');
  runAllButton.position(canvasWidth - 170, drawHeight + 13);
  runAllButton.mousePressed(runAll);
  runAllButton.size(75, 25);

  resetButton = createButton('Reset');
  resetButton.position(canvasWidth - 85, drawHeight + 13);
  resetButton.mousePressed(resetFolds);
  resetButton.size(75, 25);

  describe('Interactive visualization showing K-fold cross-validation data partitioning', LABEL);
}

function draw() {
  updateCanvasSize();
  numFolds = kSlider.value();

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
  text('K-Fold Cross-Validation Visualization', canvasWidth / 2, 10);

  // Reset text settings
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);

  // Draw the data partitioning visualization
  drawDataPartition();

  // Draw fold iterations
  drawFoldIterations();

  // Draw metrics summary
  drawMetricsSummary();

  // Control labels
  fill('black');
  noStroke();
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Number of Folds (k): ' + numFolds, 10, drawHeight + 20);

  // Current fold indicator
  fill(100);
  textSize(14);
  text(`Current Fold: ${currentFold} of ${numFolds}`, 10, drawHeight + 50);
  text(`Training samples: ${Math.floor(totalSamples * (numFolds - 1) / numFolds)} | Validation samples: ${Math.floor(totalSamples / numFolds)}`, 10, drawHeight + 75);
}

function drawDataPartition() {
  let startY = 60;
  let dataHeight = 80;
  let dataWidth = canvasWidth - 2 * margin;
  let foldWidth = dataWidth / numFolds;

  // Title
  fill(0);
  textSize(16);
  textAlign(LEFT, TOP);
  text('Data Partitioning (Total: ' + totalSamples + ' samples)', margin, startY - 25);

  // Draw all folds
  for (let i = 0; i < numFolds; i++) {
    let x = margin + i * foldWidth;
    let isValidation = (i === currentFold - 1);

    fill(isValidation ? valColor : trainColor);
    stroke(255);
    strokeWeight(2);
    rect(x, startY, foldWidth, dataHeight);

    // Label
    fill(255);
    noStroke();
    textSize(14);
    textAlign(CENTER, CENTER);
    text('Fold ' + (i + 1), x + foldWidth / 2, startY + dataHeight / 2);
  }

  // Legend
  let legendY = startY + dataHeight + 15;
  fill(trainColor);
  noStroke();
  rect(margin, legendY, 20, 15);
  fill(0);
  textSize(13);
  textAlign(LEFT, CENTER);
  text('Training Set', margin + 25, legendY + 7);

  fill(valColor);
  noStroke();
  rect(margin + 150, legendY, 20, 15);
  fill(0);
  text('Validation Set', margin + 175, legendY + 7);
}

function drawFoldIterations() {
  let startY = 200;
  let rowHeight = 50;

  // Title
  fill(0);
  textSize(16);
  textAlign(LEFT, TOP);
  text('Cross-Validation Iterations', margin, startY - 25);

  // Draw grid showing which fold is validation for each iteration
  for (let iter = 0; iter < numFolds; iter++) {
    let y = startY + iter * rowHeight;
    let foldWidth = (canvasWidth - 2 * margin - 100) / numFolds;

    // Iteration label
    fill(0);
    noStroke();
    textSize(13);
    textAlign(RIGHT, CENTER);
    text('Iter ' + (iter + 1), margin + 50, y + 20);

    // Draw folds for this iteration
    for (let f = 0; f < numFolds; f++) {
      let x = margin + 60 + f * foldWidth;
      let isVal = (f === iter);
      let isCurrentIter = (iter + 1 === currentFold);

      fill(isVal ? valColor : trainColor);
      stroke(isCurrentIter ? color(0) : color(200));
      strokeWeight(isCurrentIter ? 3 : 1);
      rect(x, y, foldWidth, 40);

      // Small label
      fill(255);
      noStroke();
      textSize(10);
      textAlign(CENTER, CENTER);
      text(isVal ? 'Val' : 'Train', x + foldWidth / 2, y + 20);
    }

    // Highlight current fold
    if (iter + 1 === currentFold) {
      noFill();
      stroke(255, 235, 59);
      strokeWeight(4);
      rect(margin + 55, y - 2, (canvasWidth - 2 * margin - 105), 44);
    }
  }
}

function drawMetricsSummary() {
  let panelX = margin;
  let panelY = drawHeight - 120;
  let panelW = canvasWidth - 2 * margin;
  let panelH = 100;

  fill(255, 255, 255, 240);
  stroke(200);
  strokeWeight(1);
  rect(panelX, panelY, panelW, panelH, 10);

  fill('black');
  noStroke();
  textAlign(LEFT, TOP);
  textSize(14);
  text('Key Concepts:', panelX + 10, panelY + 8);

  textSize(13);
  text(`• Each fold serves as validation set exactly once`, panelX + 10, panelY + 30);
  text(`• Training set uses remaining ${numFolds - 1} folds (${Math.round(((numFolds - 1) / numFolds) * 100)}% of data)`, panelX + 10, panelY + 50);
  text(`• Final metric is average across all ${numFolds} iterations`, panelX + 10, panelY + 70);
}

function nextFold() {
  currentFold++;
  if (currentFold > numFolds) {
    currentFold = 1;
  }
}

function runAll() {
  // Cycle through all folds
  currentFold = numFolds;
}

function resetFolds() {
  currentFold = 1;
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(canvasWidth, canvasHeight);
}

function updateCanvasSize() {
  const container = document.querySelector('main');
  if (container) {
    canvasWidth = container.offsetWidth;
    if (typeof kSlider !== 'undefined') {
      kSlider.size(canvasWidth - sliderLeftMargin - margin - 270);
      nextButton.position(canvasWidth - 260, drawHeight + 13);
      runAllButton.position(canvasWidth - 170, drawHeight + 13);
      resetButton.position(canvasWidth - 85, drawHeight + 13);
    }
  }
}
