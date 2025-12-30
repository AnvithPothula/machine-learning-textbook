// Entropy and Gini Impurity Comparison MicroSim
// Visualizes how entropy and Gini impurity measure dataset purity

// Canvas dimensions
let canvasWidth = 800;
let drawHeight = 600;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let margin = 40;
let sliderLeftMargin = 200;
let defaultTextSize = 16;

// Parameters
let propSlider;
let classProportion = 0.5;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.parent(document.querySelector('main'));

  // Create proportion slider
  propSlider = createSlider(0, 100, 50, 1);
  propSlider.position(sliderLeftMargin, drawHeight + 15);
  propSlider.size(canvasWidth - sliderLeftMargin - margin);

  describe('Interactive visualization comparing entropy and Gini impurity metrics for decision tree splits', LABEL);
}

function draw() {
  updateCanvasSize();
  classProportion = propSlider.value() / 100;

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
  text('Entropy vs Gini Impurity Comparison', canvasWidth / 2, 10);

  // Reset text settings
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);

  // Draw both metrics side by side
  let halfWidth = (canvasWidth - 30) / 2;
  drawEntropySection(10, 50, halfWidth, drawHeight - 170);
  drawGiniSection(halfWidth + 20, 50, halfWidth, drawHeight - 170);

  // Draw divider
  stroke(150);
  strokeWeight(2);
  line(canvasWidth / 2, 50, canvasWidth / 2, drawHeight - 120);

  // Draw comparison metrics at bottom
  drawComparisonMetrics();

  // Control labels
  fill('black');
  noStroke();
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Class 1 Proportion: ' + (classProportion * 100).toFixed(0) + '%', 10, drawHeight + 20);

  // Draw instruction
  fill(100);
  textSize(13);
  text('Move slider to see how both metrics measure impurity', 10, drawHeight + 50);
}

function drawEntropySection(x, y, w, h) {
  push();
  translate(x, y);

  // Title
  fill(33, 150, 243);
  noStroke();
  textSize(18);
  textAlign(CENTER, TOP);
  text('Entropy', w / 2, 5);

  // Formula
  textSize(13);
  fill(100);
  text('H = -Σ pᵢ log₂(pᵢ)', w / 2, 28);

  // Calculate current entropy
  let p1 = classProportion;
  let p0 = 1 - p1;
  let entropy = 0;
  if (p1 > 0 && p1 < 1) {
    entropy = -p1 * Math.log2(p1) - p0 * Math.log2(p0);
  }

  // Draw plot
  let plotX = 20;
  let plotY = 60;
  let plotW = w - 40;
  let plotH = h - 130;

  // Axes
  stroke(100);
  strokeWeight(2);
  line(plotX, plotY + plotH, plotX + plotW, plotY + plotH); // X-axis
  line(plotX, plotY, plotX, plotY + plotH); // Y-axis

  // Axis labels
  fill(0);
  noStroke();
  textSize(12);
  textAlign(CENTER, TOP);
  text('Class 1 Proportion (p)', plotX + plotW / 2, plotY + plotH + 5);

  push();
  translate(plotX - 15, plotY + plotH / 2);
  rotate(-PI / 2);
  textAlign(CENTER, CENTER);
  text('Entropy', 0, 0);
  pop();

  // Draw entropy curve
  stroke(33, 150, 243);
  strokeWeight(3);
  noFill();
  beginShape();
  for (let i = 0; i <= plotW; i += 2) {
    let p = i / plotW;
    let h_val = 0;
    if (p > 0 && p < 1) {
      h_val = -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
    }
    let px = plotX + i;
    let py = plotY + plotH - (h_val * plotH);
    vertex(px, py);
  }
  endShape();

  // Mark current value
  let currentX = plotX + classProportion * plotW;
  let currentY = plotY + plotH - (entropy * plotH);

  fill(33, 150, 243);
  stroke(255);
  strokeWeight(2);
  circle(currentX, currentY, 16);

  // Display value
  fill(33, 150, 243);
  noStroke();
  textSize(16);
  textAlign(CENTER, TOP);
  textStyle(BOLD);
  text('Entropy: ' + entropy.toFixed(3), w / 2, h - 60);
  textStyle(NORMAL);

  // Interpretation
  fill(0);
  textSize(12);
  textAlign(CENTER, TOP);
  if (entropy < 0.3) {
    text('Low impurity (pure)', w / 2, h - 35);
  } else if (entropy < 0.7) {
    text('Medium impurity', w / 2, h - 35);
  } else {
    text('High impurity (mixed)', w / 2, h - 35);
  }

  pop();
}

function drawGiniSection(x, y, w, h) {
  push();
  translate(x, y);

  // Title
  fill(76, 175, 80);
  noStroke();
  textSize(18);
  textAlign(CENTER, TOP);
  text('Gini Impurity', w / 2, 5);

  // Formula
  textSize(13);
  fill(100);
  text('Gini = 1 - Σ pᵢ²', w / 2, 28);

  // Calculate current Gini
  let p1 = classProportion;
  let p0 = 1 - p1;
  let gini = 1 - (p1 * p1 + p0 * p0);

  // Draw plot
  let plotX = 20;
  let plotY = 60;
  let plotW = w - 40;
  let plotH = h - 130;

  // Axes
  stroke(100);
  strokeWeight(2);
  line(plotX, plotY + plotH, plotX + plotW, plotY + plotH); // X-axis
  line(plotX, plotY, plotX, plotY + plotH); // Y-axis

  // Axis labels
  fill(0);
  noStroke();
  textSize(12);
  textAlign(CENTER, TOP);
  text('Class 1 Proportion (p)', plotX + plotW / 2, plotY + plotH + 5);

  push();
  translate(plotX - 15, plotY + plotH / 2);
  rotate(-PI / 2);
  textAlign(CENTER, CENTER);
  text('Gini Impurity', 0, 0);
  pop();

  // Draw Gini curve
  stroke(76, 175, 80);
  strokeWeight(3);
  noFill();
  beginShape();
  for (let i = 0; i <= plotW; i += 2) {
    let p = i / plotW;
    let g_val = 1 - (p * p + (1 - p) * (1 - p));
    let px = plotX + i;
    let py = plotY + plotH - (g_val * plotH * 2); // Scale by 2 since max Gini is 0.5
    vertex(px, py);
  }
  endShape();

  // Mark current value
  let currentX = plotX + classProportion * plotW;
  let currentY = plotY + plotH - (gini * plotH * 2);

  fill(76, 175, 80);
  stroke(255);
  strokeWeight(2);
  circle(currentX, currentY, 16);

  // Display value
  fill(76, 175, 80);
  noStroke();
  textSize(16);
  textAlign(CENTER, TOP);
  textStyle(BOLD);
  text('Gini: ' + gini.toFixed(3), w / 2, h - 60);
  textStyle(NORMAL);

  // Interpretation
  fill(0);
  textSize(12);
  textAlign(CENTER, TOP);
  if (gini < 0.15) {
    text('Low impurity (pure)', w / 2, h - 35);
  } else if (gini < 0.35) {
    text('Medium impurity', w / 2, h - 35);
  } else {
    text('High impurity (mixed)', w / 2, h - 35);
  }

  pop();
}

function drawComparisonMetrics() {
  let p1 = classProportion;
  let p0 = 1 - p1;

  let entropy = 0;
  if (p1 > 0 && p1 < 1) {
    entropy = -p1 * Math.log2(p1) - p0 * Math.log2(p0);
  }

  let gini = 1 - (p1 * p1 + p0 * p0);

  // Draw comparison panel
  let panelX = 10;
  let panelY = drawHeight - 105;
  let panelW = canvasWidth - 20;
  let panelH = 95;

  fill(255, 255, 255, 240);
  stroke(200);
  strokeWeight(1);
  rect(panelX, panelY, panelW, panelH, 10);

  fill('black');
  noStroke();
  textAlign(LEFT, TOP);
  textSize(14);
  text('Current Split Purity:', panelX + 10, panelY + 8);

  // Show class distribution
  textSize(13);
  text(`Class 0: ${(p0 * 100).toFixed(1)}% | Class 1: ${(p1 * 100).toFixed(1)}%`, panelX + 10, panelY + 30);

  // Show metrics
  text(`Entropy: ${entropy.toFixed(3)} (max = 1.0, pure = 0.0)`, panelX + 10, panelY + 50);
  text(`Gini: ${gini.toFixed(3)} (max = 0.5, pure = 0.0)`, panelX + 10, panelY + 70);

  // Show relationship
  fill(100);
  textSize(12);
  textAlign(RIGHT, TOP);
  text('Both metrics peak at p=0.5 (maximum impurity)', panelX + panelW - 10, panelY + 50);
  text('Both reach 0 at p=0 or p=1 (pure split)', panelX + panelW - 10, panelY + 70);
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(canvasWidth, canvasHeight);
}

function updateCanvasSize() {
  const container = document.querySelector('main');
  if (container) {
    canvasWidth = container.offsetWidth;
    if (typeof propSlider !== 'undefined') {
      propSlider.size(canvasWidth - sliderLeftMargin - margin);
    }
  }
}
