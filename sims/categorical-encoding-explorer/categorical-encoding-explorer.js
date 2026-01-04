// Categorical Encoding Explorer - Compare Label vs One-Hot Encoding
// Shows how different encoding methods transform categorical variables

// Canvas dimensions
let canvasWidth = 900;
let drawHeight = 680;
let controlHeight = 170;
let canvasHeight = drawHeight + controlHeight;
let margin = 20;
let sliderLeftMargin = 150;
let defaultTextSize = 16;

// Data
let sampleData = [
  { id: 1, color: 'Red', size: 'S', city: 'NYC' },
  { id: 2, color: 'Green', size: 'M', city: 'LA' },
  { id: 3, color: 'Blue', size: 'L', city: 'SF' },
  { id: 4, color: 'Red', size: 'M', city: 'NYC' },
  { id: 5, color: 'Green', size: 'S', city: 'LA' }
];

// Parameters
let dropFirst = true;
let exampleDataset = 'colors';
let dropFirstCheckbox, exampleSelector, addRowBtn, resetBtn;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.parent(document.querySelector('main'));

  // Example dataset selector
  exampleSelector = createSelect();
  exampleSelector.option('Color Data');
  exampleSelector.option('Iris Species');
  exampleSelector.option('Car Types');
  exampleSelector.selected('Color Data');
  exampleSelector.position(sliderLeftMargin, drawHeight + 15);
  exampleSelector.changed(() => {
    loadExampleDataset(exampleSelector.value());
  });

  // Drop first checkbox
  dropFirstCheckbox = createCheckbox('One-Hot drop_first', dropFirst);
  dropFirstCheckbox.position(sliderLeftMargin, drawHeight + 50);
  dropFirstCheckbox.changed(() => {
    dropFirst = dropFirstCheckbox.checked();
  });

  // Add row button
  addRowBtn = createButton('Add Row');
  addRowBtn.position(canvasWidth - 280, drawHeight + 13);
  addRowBtn.mousePressed(addRow);
  addRowBtn.size(80, 25);

  // Reset button
  resetBtn = createButton('Reset');
  resetBtn.position(canvasWidth - 190, drawHeight + 13);
  resetBtn.mousePressed(resetData);
  resetBtn.size(80, 25);

  describe('Interactive visualization comparing label encoding and one-hot encoding for categorical variables', LABEL);
}

function loadExampleDataset(dataset) {
  if (dataset === 'Iris Species') {
    sampleData = [
      { id: 1, species: 'setosa', petal: 'small', habitat: 'wetland' },
      { id: 2, species: 'versicolor', petal: 'medium', habitat: 'meadow' },
      { id: 3, species: 'virginica', petal: 'large', habitat: 'woodland' },
      { id: 4, species: 'setosa', petal: 'small', habitat: 'wetland' },
      { id: 5, species: 'versicolor', petal: 'medium', habitat: 'meadow' }
    ];
  } else if (dataset === 'Car Types') {
    sampleData = [
      { id: 1, make: 'Toyota', type: 'sedan', fuel: 'gas' },
      { id: 2, make: 'Tesla', type: 'SUV', fuel: 'electric' },
      { id: 3, make: 'Ford', type: 'truck', fuel: 'diesel' },
      { id: 4, make: 'Toyota', type: 'sedan', fuel: 'hybrid' },
      { id: 5, make: 'Tesla', type: 'sedan', fuel: 'electric' }
    ];
  } else {
    sampleData = [
      { id: 1, color: 'Red', size: 'S', city: 'NYC' },
      { id: 2, color: 'Green', size: 'M', city: 'LA' },
      { id: 3, color: 'Blue', size: 'L', city: 'SF' },
      { id: 4, color: 'Red', size: 'M', city: 'NYC' },
      { id: 5, color: 'Green', size: 'S', city: 'LA' }
    ];
  }
}

function addRow() {
  const lastRow = sampleData[sampleData.length - 1];
  const cols = Object.keys(sampleData[0]).filter(col => col !== 'id');
  const newRow = { id: lastRow.id + 1 };

  for (let col of cols) {
    const uniqueValues = [...new Set(sampleData.map(row => row[col]))];
    newRow[col] = random(uniqueValues);
  }

  sampleData.push(newRow);
}

function resetData() {
  loadExampleDataset(exampleSelector.value());
  dropFirst = true;
  dropFirstCheckbox.checked(true);
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
  text('Categorical Encoding Comparison', canvasWidth / 2, 10);

  // Reset text settings
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);

  // Draw three panels side by side
  let panelWidth = (canvasWidth - 40) / 3;
  drawOriginalTable(10, 50, panelWidth, 280);
  drawLabelEncoding(panelWidth + 15, 50, panelWidth, 280);
  drawOneHotEncoding(2 * panelWidth + 20, 50, panelWidth, 280);

  // Draw comparison panel
  drawComparisonPanel(10, 350, canvasWidth - 20, 280);

  // Control labels
  fill('black');
  noStroke();
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Example Dataset:', 10, drawHeight + 20);
  text('Options:', 10, drawHeight + 55);

  // Draw instruction
  fill(100);
  textSize(13);
  text('Compare different categorical encoding methods', 10, drawHeight + 90);
}

function drawOriginalTable(x, y, w, h) {
  push();
  translate(x, y);

  // Title
  fill(63, 81, 181);
  noStroke();
  textSize(14);
  textAlign(CENTER, TOP);
  textStyle(BOLD);
  text('Original Data', w / 2, 5);

  // Get columns
  const cols = Object.keys(sampleData[0]);
  const cellHeight = 25;
  const headerHeight = 30;
  const colWidth = (w - 20) / cols.length;

  // Header
  fill(63, 81, 181);
  rect(10, 30, w - 20, headerHeight);

  fill(255);
  textAlign(CENTER, CENTER);
  textSize(11);
  for (let i = 0; i < cols.length; i++) {
    text(cols[i], 10 + i * colWidth + colWidth / 2, 30 + headerHeight / 2);
  }

  // Rows
  textStyle(NORMAL);
  textSize(10);
  for (let row = 0; row < min(sampleData.length, 7); row++) {
    const rowY = 30 + headerHeight + row * cellHeight;

    fill(row % 2 === 0 ? 250 : 240);
    noStroke();
    rect(10, rowY, w - 20, cellHeight);

    fill(0);
    for (let col = 0; col < cols.length; col++) {
      const value = sampleData[row][cols[col]];
      text(value, 10 + col * colWidth + colWidth / 2, rowY + cellHeight / 2);
    }
  }

  // Border
  stroke(200);
  noFill();
  rect(10, 30, w - 20, headerHeight + min(sampleData.length, 7) * cellHeight);

  pop();
}

function drawLabelEncoding(x, y, w, h) {
  push();
  translate(x, y);

  // Title
  fill(233, 30, 99);
  noStroke();
  textSize(14);
  textAlign(CENTER, TOP);
  textStyle(BOLD);
  text('Label Encoding', w / 2, 5);

  const catCols = Object.keys(sampleData[0]).filter(col => col !== 'id');
  const cellHeight = 25;
  const headerHeight = 30;
  const colWidth = (w - 20) / (catCols.length + 1);

  // Create mappings
  const mappings = {};
  for (let col of catCols) {
    const uniqueValues = [...new Set(sampleData.map(row => row[col]))];
    mappings[col] = Object.fromEntries(uniqueValues.map((val, idx) => [val, idx]));
  }

  // Header
  fill(233, 30, 99);
  rect(10, 30, w - 20, headerHeight);

  fill(255);
  textAlign(CENTER, CENTER);
  textSize(11);
  textStyle(BOLD);
  text('id', 10 + colWidth / 2, 30 + headerHeight / 2);
  for (let i = 0; i < catCols.length; i++) {
    text(catCols[i], 10 + (i + 1) * colWidth + colWidth / 2, 30 + headerHeight / 2);
  }

  // Rows
  textStyle(NORMAL);
  textSize(10);
  for (let row = 0; row < min(sampleData.length, 7); row++) {
    const rowY = 30 + headerHeight + row * cellHeight;

    fill(row % 2 === 0 ? 250 : 240);
    noStroke();
    rect(10, rowY, w - 20, cellHeight);

    fill(0);
    text(sampleData[row].id, 10 + colWidth / 2, rowY + cellHeight / 2);

    for (let col = 0; col < catCols.length; col++) {
      const originalValue = sampleData[row][catCols[col]];
      const labelValue = mappings[catCols[col]][originalValue];
      text(labelValue, 10 + (col + 1) * colWidth + colWidth / 2, rowY + cellHeight / 2);
    }
  }

  // Border
  stroke(200);
  noFill();
  rect(10, 30, w - 20, headerHeight + min(sampleData.length, 7) * cellHeight);

  pop();
}

function drawOneHotEncoding(x, y, w, h) {
  push();
  translate(x, y);

  // Title
  fill(0, 150, 136);
  noStroke();
  textSize(14);
  textAlign(CENTER, TOP);
  textStyle(BOLD);
  text('One-Hot Encoding', w / 2, 5);

  const catCols = Object.keys(sampleData[0]).filter(col => col !== 'id');
  const cellHeight = 25;
  const headerHeight = 30;

  // Create one-hot columns
  const oneHotCols = ['id'];
  const mappings = {};

  for (let col of catCols) {
    const uniqueValues = [...new Set(sampleData.map(row => row[col]))].sort();
    mappings[col] = uniqueValues;
    const start = dropFirst ? 1 : 0;
    for (let i = start; i < uniqueValues.length; i++) {
      oneHotCols.push(`${col}_${uniqueValues[i]}`);
    }
  }

  const colWidth = (w - 20) / min(oneHotCols.length, 7);

  // Header
  fill(0, 150, 136);
  rect(10, 30, w - 20, headerHeight);

  fill(255);
  textAlign(CENTER, CENTER);
  textSize(9);
  textStyle(BOLD);
  for (let i = 0; i < min(oneHotCols.length, 7); i++) {
    text(oneHotCols[i], 10 + i * colWidth + colWidth / 2, 30 + headerHeight / 2);
  }

  // Rows
  textStyle(NORMAL);
  textSize(10);
  for (let row = 0; row < min(sampleData.length, 7); row++) {
    const rowY = 30 + headerHeight + row * cellHeight;

    fill(row % 2 === 0 ? 250 : 240);
    noStroke();
    rect(10, rowY, w - 20, cellHeight);

    fill(0);
    text(sampleData[row].id, 10 + colWidth / 2, rowY + cellHeight / 2);

    let colIdx = 1;
    for (let catCol of catCols) {
      const originalValue = sampleData[row][catCol];
      const uniqueValues = mappings[catCol];
      const start = dropFirst ? 1 : 0;

      for (let i = start; i < uniqueValues.length && colIdx < 7; i++) {
        const isMatch = uniqueValues[i] === originalValue;
        const cellX = 10 + colIdx * colWidth + colWidth / 2;

        if (isMatch) {
          fill(76, 175, 80);
          noStroke();
          circle(cellX, rowY + cellHeight / 2, 14);
          fill(255);
          textStyle(BOLD);
        } else {
          fill(0);
          textStyle(NORMAL);
        }

        text(isMatch ? '1' : '0', cellX, rowY + cellHeight / 2);
        colIdx++;
      }
    }
  }

  // Border
  stroke(200);
  noFill();
  rect(10, 30, w - 20, headerHeight + min(sampleData.length, 7) * cellHeight);

  pop();
}

function drawComparisonPanel(x, y, w, h) {
  push();
  translate(x, y);

  fill(255, 255, 255, 240);
  stroke(200);
  strokeWeight(1);
  rect(0, 0, w, h, 10);

  fill('black');
  noStroke();
  textAlign(LEFT, TOP);
  textSize(14);
  textStyle(BOLD);
  text('Comparison: Label Encoding vs One-Hot Encoding', 15, 10);

  // Calculate dimensions
  const catCols = Object.keys(sampleData[0]).filter(col => col !== 'id');
  let oneHotCols = 1;
  for (let col of catCols) {
    const uniqueValues = [...new Set(sampleData.map(row => row[col]))];
    oneHotCols += dropFirst ? uniqueValues.length - 1 : uniqueValues.length;
  }

  textStyle(NORMAL);
  textSize(12);
  text(`Original columns: ${catCols.length + 1}`, 20, 45);
  text(`Label encoding columns: ${catCols.length + 1}`, 20, 70);
  text(`One-hot encoding columns: ${oneHotCols}`, 20, 95);

  // Pros and cons
  textStyle(BOLD);
  textSize(13);
  text('Label Encoding', 30, 130);
  text('One-Hot Encoding', w / 2 + 20, 130);

  textStyle(NORMAL);
  textSize(11);

  // Label encoding pros/cons
  fill(0, 128, 0);
  text('+ Memory efficient', 30, 155);
  text('+ Single column per feature', 30, 175);
  fill(200, 0, 0);
  text('- Implies false ordering', 30, 195);
  text('- Not for nominal data', 30, 215);

  // One-hot pros/cons
  fill(0, 128, 0);
  text('+ No artificial ordering', w / 2 + 20, 155);
  text('+ Works with all algorithms', w / 2 + 20, 175);
  fill(200, 0, 0);
  text('- Increases dimensionality', w / 2 + 20, 195);
  text('- Sparse for many categories', w / 2 + 20, 215);

  fill(0);
  textStyle(BOLD);
  textSize(12);
  text('When to use:', 20, 240);

  textStyle(NORMAL);
  textSize(11);
  text('Label: Tree models, ordinal variables, target encoding', 20, 260);
  text('One-Hot: Linear models, neural networks, nominal variables', 20, 275);

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
    if (typeof exampleSelector !== 'undefined') {
      exampleSelector.position(sliderLeftMargin, drawHeight + 15);
      dropFirstCheckbox.position(sliderLeftMargin, drawHeight + 50);
      addRowBtn.position(canvasWidth - 280, drawHeight + 13);
      resetBtn.position(canvasWidth - 190, drawHeight + 13);
    }
  }
}
