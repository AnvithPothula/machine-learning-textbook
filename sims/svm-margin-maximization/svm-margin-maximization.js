// SVM Margin Maximization MicroSim
// Visualizes decision boundary, margins, and support vectors

// Canvas dimensions
let canvasWidth = 600;
let drawHeight = 500;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let margin = 40;
let sliderLeftMargin = 170;
let defaultTextSize = 16;

// Simulation parameters
let marginSlider;
let marginWidth = 140; // Width of the margin in pixels
let showMarginsCheckbox;
let showMargins = true;

// Data points
let class1Points = [];
let class2Points = [];
let supportVectors = [];

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.parent(document.querySelector('main'));

  // Create margin width slider
  marginSlider = createSlider(60, 200, marginWidth, 10);
  marginSlider.position(sliderLeftMargin, drawHeight + 15);
  marginSlider.size(canvasWidth - sliderLeftMargin - margin);

  // Create checkbox for showing margins
  showMarginsCheckbox = createCheckbox('Show Margin Boundaries', showMargins);
  showMarginsCheckbox.position(10, drawHeight + 50);
  showMarginsCheckbox.changed(() => {
    showMargins = showMarginsCheckbox.checked();
  });

  // Generate data points
  generateDataPoints();

  describe('Interactive visualization of SVM margin maximization showing decision boundary, margins, and support vectors', LABEL);
}

function draw() {
  updateCanvasSize();
  marginWidth = marginSlider.value();

  // Update support vectors based on margin width
  updateSupportVectors();

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
  text('SVM Margin Maximization', canvasWidth / 2, 10);

  // Reset text settings
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);

  // Decision boundary is vertical at center
  let boundaryX = canvasWidth / 2;

  // Draw margin region (shaded)
  if (showMargins) {
    fill(227, 242, 253, 128); // Light blue with transparency
    noStroke();
    rect(boundaryX - marginWidth/2, margin + 20, marginWidth, drawHeight - margin - 30);
  }

  // Draw margin boundaries (dashed lines)
  if (showMargins) {
    stroke(100);
    strokeWeight(2);
    drawingContext.setLineDash([5, 5]);
    line(boundaryX - marginWidth/2, margin + 20, boundaryX - marginWidth/2, drawHeight - margin);
    line(boundaryX + marginWidth/2, margin + 20, boundaryX + marginWidth/2, drawHeight - margin);
    drawingContext.setLineDash([]);
  }

  // Draw decision boundary (solid line)
  stroke(33, 150, 243); // Blue
  strokeWeight(3);
  line(boundaryX, margin + 20, boundaryX, drawHeight - margin);

  // Label decision boundary
  fill(33, 150, 243);
  noStroke();
  textAlign(CENTER);
  textSize(13);
  text('Decision\nBoundary', boundaryX + 5, drawHeight / 2);

  // Draw all data points
  drawDataPoints();

  // Draw support vectors (highlighted)
  drawSupportVectors();

  // Draw margin width annotation
  drawMarginAnnotation(boundaryX);

  // Draw class labels
  fill(244, 67, 54);
  noStroke();
  textAlign(CENTER);
  textSize(15);
  text('Class -1', boundaryX - marginWidth/2 - 60, 50);

  fill(33, 150, 243);
  text('Class +1', boundaryX + marginWidth/2 + 60, 50);

  // Draw legend
  drawLegend();

  // Control labels
  fill('black');
  noStroke();
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Margin Width: ' + marginWidth + ' px', 10, drawHeight + 20);
}

function generateDataPoints() {
  // Generate Class -1 points (left side, red circles)
  class1Points = [
    {x: 80, y: 100},
    {x: 120, y: 150},
    {x: 90, y: 200},
    {x: 110, y: 250},
    {x: 70, y: 300},
    {x: 95, y: 350},
    {x: 130, y: 400}
  ];

  // Generate Class +1 points (right side, blue squares)
  class2Points = [
    {x: 470, y: 100},
    {x: 510, y: 150},
    {x: 490, y: 200},
    {x: 500, y: 250},
    {x: 520, y: 300},
    {x: 495, y: 350},
    {x: 485, y: 400}
  ];
}

function updateSupportVectors() {
  // Support vectors are points on the margin boundaries
  let boundaryX = canvasWidth / 2;
  let leftMargin = boundaryX - marginWidth/2;
  let rightMargin = boundaryX + marginWidth/2;

  supportVectors = [
    {x: leftMargin, y: 120, class: -1},
    {x: leftMargin, y: 280, class: -1},
    {x: rightMargin, y: 155, class: 1},
    {x: rightMargin, y: 245, class: 1}
  ];
}

function drawDataPoints() {
  // Draw Class -1 points (circles)
  fill(244, 67, 54); // Red
  noStroke();
  for (let p of class1Points) {
    circle(p.x, p.y, 16);
  }

  // Draw Class +1 points (squares)
  fill(33, 150, 243); // Blue
  noStroke();
  for (let p of class2Points) {
    rect(p.x - 8, p.y - 8, 16, 16);
  }
}

function drawSupportVectors() {
  for (let sv of supportVectors) {
    if (sv.class === -1) {
      // Class -1 support vector (circle with thick border)
      fill(244, 67, 54);
      stroke(0);
      strokeWeight(3);
      circle(sv.x, sv.y, 20);
    } else {
      // Class +1 support vector (square with thick border)
      fill(33, 150, 243);
      stroke(0);
      strokeWeight(3);
      rect(sv.x - 10, sv.y - 10, 20, 20);
    }
  }
}

function drawMarginAnnotation(boundaryX) {
  // Draw arrow showing margin width
  let annotY = drawHeight - 15;
  stroke(255, 152, 0); // Orange
  strokeWeight(2);
  line(boundaryX - marginWidth/2, annotY, boundaryX + marginWidth/2, annotY);

  // End caps
  line(boundaryX - marginWidth/2, annotY - 5, boundaryX - marginWidth/2, annotY + 5);
  line(boundaryX + marginWidth/2, annotY - 5, boundaryX + marginWidth/2, annotY + 5);

  // Label
  fill(255, 152, 0);
  noStroke();
  textAlign(CENTER);
  textSize(13);
  text('Margin = 2/||w||', boundaryX, annotY - 12);
}

function drawLegend() {
  let legendX = 10;
  let legendY = 80;

  // Support vector indicator
  fill(200);
  stroke(0);
  strokeWeight(3);
  circle(legendX + 8, legendY, 16);

  fill('black');
  noStroke();
  textAlign(LEFT, CENTER);
  textSize(12);
  text('= Support Vector', legendX + 25, legendY);

  // Key insights panel
  let panelX = canvasWidth - 280;
  let panelY = 80;
  let panelW = 270;
  let panelH = 135;

  fill(255, 255, 255, 240);
  stroke(200);
  strokeWeight(1);
  rect(panelX, panelY, panelW, panelH, 10);

  fill('black');
  noStroke();
  textAlign(LEFT, TOP);
  textSize(12);
  text('Key Concepts:', panelX + 10, panelY + 5);
  text('• Maximum Margin: SVM finds the', panelX + 10, panelY + 22);
  text('  widest street between classes', panelX + 10, panelY + 37);
  text('• Support Vectors: Points on margin', panelX + 10, panelY + 54);
  text('  boundaries (thick borders) that', panelX + 10, panelY + 69);
  text('  define the decision boundary', panelX + 10, panelY + 84);
  text('• Optimal Separation: Only support', panelX + 10, panelY + 101);
  text('  vectors matter for the solution', panelX + 10, panelY + 116);
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(canvasWidth, canvasHeight);
}

function updateCanvasSize() {
  const container = document.querySelector('main');
  if (container) {
    let newWidth = container.offsetWidth;
    if (newWidth !== canvasWidth) {
      // Adjust data points proportionally
      let ratio = newWidth / canvasWidth;
      for (let p of class1Points) {
        p.x *= ratio;
      }
      for (let p of class2Points) {
        p.x *= ratio;
      }
      canvasWidth = newWidth;
    }

    if (typeof marginSlider !== 'undefined') {
      marginSlider.size(canvasWidth - sliderLeftMargin - margin);
    }
  }
}
