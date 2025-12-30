// Feature Scaling Visualizer - Compare min-max and z-score scaling
// Shows how different scaling methods transform data distributions

// Canvas dimensions
let canvasWidth = 900;
let drawHeight = 600;
let controlHeight = 120;
let canvasHeight = drawHeight + controlHeight;
let margin = 20;
let sliderLeftMargin = 150;
let defaultTextSize = 16;

// Data
let data = [];
let scaledMinMax = [];
let scaledZScore = [];

// Parameters
let sampleSize = 500;
let dataMean = 100;
let dataStd = 15;
let distributionType = 'normal';

// Controls
let sampleSlider, meanSlider, stdSlider;
let distSelector, addOutliersBtn, resetBtn;

// Statistics
let stats = {
  original: { min: 0, max: 0, mean: 0, std: 0 },
  minMax: { min: 0, max: 0, mean: 0, std: 0 },
  zScore: { min: 0, max: 0, mean: 0, std: 0 }
};

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.parent(document.querySelector('main'));

  // Distribution selector
  distSelector = createSelect();
  distSelector.option('Normal');
  distSelector.option('Skewed');
  distSelector.option('Bimodal');
  distSelector.selected('Normal');
  distSelector.position(sliderLeftMargin, drawHeight + 15);
  distSelector.changed(() => {
    distributionType = distSelector.value().toLowerCase();
    generateData();
    computeScaling();
  });

  // Sample size slider
  sampleSlider = createSlider(100, 2000, sampleSize, 100);
  sampleSlider.position(sliderLeftMargin, drawHeight + 50);
  sampleSlider.size(200);

  // Mean slider
  meanSlider = createSlider(50, 150, dataMean, 5);
  meanSlider.position(sliderLeftMargin, drawHeight + 85);
  meanSlider.size(100);

  // Std slider
  stdSlider = createSlider(5, 50, dataStd, 1);
  stdSlider.position(sliderLeftMargin + 120, drawHeight + 85);
  stdSlider.size(100);

  // Add outliers button
  addOutliersBtn = createButton('Add Outliers');
  addOutliersBtn.position(canvasWidth - 280, drawHeight + 13);
  addOutliersBtn.mousePressed(addOutliers);
  addOutliersBtn.size(100, 25);

  // Reset button
  resetBtn = createButton('Reset');
  resetBtn.position(canvasWidth - 170, drawHeight + 13);
  resetBtn.mousePressed(resetData);
  resetBtn.size(80, 25);

  // Generate initial data
  generateData();
  computeScaling();

  describe('Interactive visualization comparing Min-Max scaling and Z-Score standardization', LABEL);
}

function generateData() {
  data = [];

  switch(distributionType) {
    case 'normal':
      for (let i = 0; i < sampleSize; i++) {
        data.push(randomGaussian(dataMean, dataStd));
      }
      break;
    case 'skewed':
      for (let i = 0; i < sampleSize; i++) {
        const val = abs(randomGaussian(0, dataStd));
        data.push(dataMean + val);
      }
      break;
    case 'bimodal':
      for (let i = 0; i < sampleSize / 2; i++) {
        data.push(randomGaussian(dataMean - 20, dataStd / 2));
        data.push(randomGaussian(dataMean + 20, dataStd / 2));
      }
      break;
  }
}

function addOutliers() {
  const numOutliers = Math.floor(sampleSize * 0.05);
  const dataMin = min(data);
  const dataMax = max(data);
  for (let i = 0; i < numOutliers; i++) {
    data.push(random(dataMin - 50, dataMax + 50));
  }
  computeScaling();
}

function resetData() {
  sampleSize = 500;
  dataMean = 100;
  dataStd = 15;
  distributionType = 'normal';

  sampleSlider.value(500);
  meanSlider.value(100);
  stdSlider.value(15);
  distSelector.selected('Normal');

  generateData();
  computeScaling();
}

function computeScaling() {
  // Compute statistics for original data
  const dataMin = min(data);
  const dataMax = max(data);
  const dataMean = data.reduce((a, b) => a + b, 0) / data.length;
  const variance = data.reduce((sum, x) => sum + pow(x - dataMean, 2), 0) / data.length;
  const dataStd = sqrt(variance);

  stats.original = { min: dataMin, max: dataMax, mean: dataMean, std: dataStd };

  // Min-Max scaling
  scaledMinMax = data.map(x => (x - dataMin) / (dataMax - dataMin));
  const minMaxMean = scaledMinMax.reduce((a, b) => a + b, 0) / scaledMinMax.length;
  const minMaxVar = scaledMinMax.reduce((sum, x) => sum + pow(x - minMaxMean, 2), 0) / scaledMinMax.length;
  stats.minMax = {
    min: min(scaledMinMax),
    max: max(scaledMinMax),
    mean: minMaxMean,
    std: sqrt(minMaxVar)
  };

  // Z-Score standardization
  scaledZScore = data.map(x => (x - dataMean) / dataStd);
  const zScoreMean = scaledZScore.reduce((a, b) => a + b, 0) / scaledZScore.length;
  const zScoreVar = scaledZScore.reduce((sum, x) => sum + pow(x - zScoreMean, 2), 0) / scaledZScore.length;
  stats.zScore = {
    min: min(scaledZScore),
    max: max(scaledZScore),
    mean: zScoreMean,
    std: sqrt(zScoreVar)
  };
}

function draw() {
  updateCanvasSize();

  // Update parameters from sliders
  sampleSize = sampleSlider.value();
  dataMean = meanSlider.value();
  dataStd = stdSlider.value();

  // Check if parameters changed
  if (frameCount % 30 === 0 && frameCount > 30) {
    const oldSize = data.length;
    if (oldSize !== sampleSize) {
      generateData();
      computeScaling();
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
  text('Feature Scaling Comparison', canvasWidth / 2, 10);

  // Reset text settings
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);

  // Draw three panels side by side
  let panelWidth = (canvasWidth - 40) / 3;
  let panelHeight = drawHeight - 60;

  drawScalingPanel(10, 50, panelWidth, panelHeight, 'Original Data', data, stats.original, color(63, 81, 181));
  drawScalingPanel(panelWidth + 15, 50, panelWidth, panelHeight, 'Min-Max [0,1]', scaledMinMax, stats.minMax, color(233, 30, 99));
  drawScalingPanel(2 * panelWidth + 20, 50, panelWidth, panelHeight, 'Z-Score', scaledZScore, stats.zScore, color(0, 150, 136));

  // Control labels
  fill('black');
  noStroke();
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Distribution:', 10, drawHeight + 20);
  text('Samples: ' + sampleSize, 10, drawHeight + 55);
  text('Mean: ' + dataMean, 10, drawHeight + 90);
  text('Std: ' + dataStd, 130, drawHeight + 90);

  // Draw instruction
  fill(100);
  textSize(13);
  text('Adjust parameters to see how scaling methods transform distributions', 10, drawHeight + 110);
}

function drawScalingPanel(x, y, w, h, title, dataset, statistics, col) {
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
  textSize(14);
  textAlign(CENTER, TOP);
  textStyle(BOLD);
  text(title, w / 2, 8);

  // Draw histogram
  const histY = 35;
  const histHeight = h * 0.5;
  drawHistogram(10, histY, w - 20, histHeight, dataset, col);

  // Draw box plot
  const boxY = histY + histHeight + 20;
  const boxHeight = 80;
  drawBoxPlot(10, boxY, w - 20, boxHeight, dataset, col);

  // Draw statistics
  const statsY = boxY + boxHeight + 15;
  fill(0);
  noStroke();
  textAlign(LEFT, TOP);
  textSize(11);
  textStyle(NORMAL);
  text('Mean: ' + statistics.mean.toFixed(3), 15, statsY);
  text('Std: ' + statistics.std.toFixed(3), 15, statsY + 18);
  text('Min: ' + statistics.min.toFixed(3), 15, statsY + 36);
  text('Max: ' + statistics.max.toFixed(3), 15, statsY + 54);

  pop();
}

function drawHistogram(x, y, w, h, dataset, col) {
  const bins = 25;
  const dataMin = min(dataset);
  const dataMax = max(dataset);
  const binWidth = (dataMax - dataMin) / bins;

  // Count frequencies
  const frequencies = new Array(bins).fill(0);
  for (let val of dataset) {
    const binIndex = Math.min(Math.floor((val - dataMin) / binWidth), bins - 1);
    frequencies[binIndex]++;
  }

  const maxFreq = max(frequencies);
  const barWidth = w / bins;

  // Draw bars
  fill(red(col), green(col), blue(col), 180);
  stroke(red(col), green(col), blue(col));
  strokeWeight(1);
  for (let i = 0; i < bins; i++) {
    const barHeight = (frequencies[i] / maxFreq) * h;
    rect(x + i * barWidth, y + h - barHeight, barWidth - 1, barHeight);
  }

  // Border
  stroke(150);
  noFill();
  rect(x, y, w, h);
}

function drawBoxPlot(x, y, w, h, dataset, col) {
  const sorted = [...dataset].sort((a, b) => a - b);
  const n = sorted.length;

  const dataMin = sorted[0];
  const dataMax = sorted[n - 1];
  const q1 = sorted[Math.floor(n * 0.25)];
  const median = sorted[Math.floor(n * 0.5)];
  const q3 = sorted[Math.floor(n * 0.75)];

  const iqr = q3 - q1;
  const lowerWhisker = max(dataMin, q1 - 1.5 * iqr);
  const upperWhisker = min(dataMax, q3 + 1.5 * iqr);

  // Map to coordinates
  const range = dataMax - dataMin || 1;
  const mapValue = (val) => x + ((val - dataMin) / range) * w;

  const centerY = y + h / 2;
  const boxHeight = h * 0.5;

  // Draw whiskers
  stroke(100);
  strokeWeight(1);
  line(mapValue(lowerWhisker), centerY, mapValue(q1), centerY);
  line(mapValue(q3), centerY, mapValue(upperWhisker), centerY);
  line(mapValue(lowerWhisker), centerY - boxHeight/2, mapValue(lowerWhisker), centerY + boxHeight/2);
  line(mapValue(upperWhisker), centerY - boxHeight/2, mapValue(upperWhisker), centerY + boxHeight/2);

  // Draw box
  fill(red(col), green(col), blue(col), 150);
  stroke(red(col), green(col), blue(col));
  strokeWeight(2);
  rect(mapValue(q1), centerY - boxHeight/2, mapValue(q3) - mapValue(q1), boxHeight);

  // Draw median
  stroke(255, 87, 34);
  strokeWeight(3);
  line(mapValue(median), centerY - boxHeight/2, mapValue(median), centerY + boxHeight/2);

  // Draw outliers
  fill(255, 87, 34);
  noStroke();
  for (let val of sorted) {
    if (val < lowerWhisker || val > upperWhisker) {
      circle(mapValue(val), centerY, 5);
    }
  }

  strokeWeight(1);
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(canvasWidth, canvasHeight);
}

function updateCanvasSize() {
  const container = document.querySelector('main');
  if (container) {
    canvasWidth = container.offsetWidth;
    if (typeof distSelector !== 'undefined') {
      distSelector.position(sliderLeftMargin, drawHeight + 15);
      sampleSlider.position(sliderLeftMargin, drawHeight + 50);
      meanSlider.position(sliderLeftMargin, drawHeight + 85);
      stdSlider.position(sliderLeftMargin + 120, drawHeight + 85);
      addOutliersBtn.position(canvasWidth - 280, drawHeight + 13);
      resetBtn.position(canvasWidth - 170, drawHeight + 13);
    }
  }
}
