// Ridge Regression Geometry MicroSim
// Visualizes L2 regularization constraint and Ridge solution

// Canvas dimensions
let canvasWidth = 500;
let drawHeight = 500;
let controlHeight = 80;
let canvasHeight = drawHeight + controlHeight;
let margin = 40;
let sliderLeftMargin = 170;
let defaultTextSize = 16;

// Simulation parameters
let lambdaSlider;
let lambda = 0.3; // Regularization strength
let showContoursCheckbox;
let showContours = true;

// Coordinate system
let centerX, centerY;
let scale = 120; // pixels per unit in beta space

// OLS solution (unconstrained minimum)
let olsBeta1 = 1.4;
let olsBeta2 = 1.0;

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(canvasWidth, canvasHeight);
  canvas.parent(document.querySelector('main'));

  centerX = canvasWidth / 2;
  centerY = drawHeight / 2;

  // Create lambda slider
  lambdaSlider = createSlider(0, 1, lambda, 0.01);
  lambdaSlider.position(sliderLeftMargin, drawHeight + 15);
  lambdaSlider.size(canvasWidth - sliderLeftMargin - margin);

  // Create checkbox for showing contours
  showContoursCheckbox = createCheckbox('Show Error Contours', showContours);
  showContoursCheckbox.position(10, drawHeight + 50);
  showContoursCheckbox.changed(() => {
    showContours = showContoursCheckbox.checked();
  });

  describe('Interactive visualization of Ridge regression L2 regularization geometry showing constraint circle and solution', LABEL);
}

function draw() {
  updateCanvasSize();
  lambda = lambdaSlider.value();

  centerX = canvasWidth / 2;
  centerY = drawHeight / 2;

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
  text('Ridge Regression Geometry (L2 Regularization)', canvasWidth / 2, 10);

  // Reset text settings
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);

  // Draw coordinate axes
  drawAxes();

  // Calculate L2 constraint radius from lambda
  // As lambda increases, constraint gets tighter (smaller radius)
  let constraintRadius = scale * (1.2 - lambda * 0.8); // radius in pixels

  // Draw L2 constraint circle
  push();
  translate(centerX, centerY);
  fill(33, 150, 243, 50); // Blue with transparency
  stroke(33, 150, 243);
  strokeWeight(3);
  circle(0, 0, constraintRadius * 2);

  // Label the constraint
  fill(33, 150, 243);
  noStroke();
  textAlign(CENTER);
  textSize(14);
  text('β₁² + β₂² ≤ t', 0, constraintRadius + 20);
  pop();

  // Draw error contours (ellipses centered at OLS solution)
  if (showContours) {
    drawErrorContours();
  }

  // Draw OLS solution
  push();
  translate(centerX, centerY);
  fill(244, 67, 54); // Red
  noStroke();
  circle(olsBeta1 * scale, -olsBeta2 * scale, 12);
  fill(244, 67, 54);
  textAlign(LEFT);
  textSize(14);
  text('OLS', olsBeta1 * scale + 10, -olsBeta2 * scale);
  pop();

  // Calculate Ridge solution (point where error contour touches L2 circle)
  // Ridge solution shrinks OLS toward origin
  let olsDistance = sqrt(olsBeta1 * olsBeta1 + olsBeta2 * olsBeta2);
  let shrinkFactor = (constraintRadius / scale) / olsDistance;
  if (shrinkFactor > 1) shrinkFactor = 1; // Don't expand beyond OLS

  let ridgeBeta1 = olsBeta1 * shrinkFactor;
  let ridgeBeta2 = olsBeta2 * shrinkFactor;

  // Draw Ridge solution
  push();
  translate(centerX, centerY);
  fill(33, 150, 243); // Blue
  noStroke();
  circle(ridgeBeta1 * scale, -ridgeBeta2 * scale, 14);
  fill(33, 150, 243);
  textAlign(LEFT);
  textSize(14);
  fontWeight = 'bold';
  text('Ridge', ridgeBeta1 * scale + 10, -ridgeBeta2 * scale);
  pop();

  // Draw shrinkage arrow
  if (shrinkFactor < 0.99) {
    push();
    translate(centerX, centerY);
    stroke(100);
    strokeWeight(2);
    drawingContext.setLineDash([5, 5]);
    line(olsBeta1 * scale, -olsBeta2 * scale, ridgeBeta1 * scale, -ridgeBeta2 * scale);
    drawingContext.setLineDash([]);

    // Arrow head
    let angle = atan2(-(ridgeBeta2 - olsBeta2), ridgeBeta1 - olsBeta1);
    let arrowSize = 10;
    push();
    translate(ridgeBeta1 * scale, -ridgeBeta2 * scale);
    rotate(angle);
    fill(100);
    noStroke();
    triangle(0, 0, -arrowSize, -arrowSize/2, -arrowSize, arrowSize/2);
    pop();

    // Label
    fill(100);
    noStroke();
    textAlign(CENTER);
    textSize(12);
    let midX = (olsBeta1 + ridgeBeta1) / 2 * scale;
    let midY = -(olsBeta2 + ridgeBeta2) / 2 * scale;
    text('Shrinkage', midX, midY - 10);
    pop();
  }

  // Draw key insights panel
  drawKeyInsights(constraintRadius / scale, shrinkFactor);

  // Control labels
  fill('black');
  noStroke();
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Regularization λ: ' + lambda.toFixed(2), 10, drawHeight + 20);
}

function drawAxes() {
  push();
  translate(centerX, centerY);

  // Axes
  stroke(100);
  strokeWeight(1);
  line(-canvasWidth/2 + margin, 0, canvasWidth/2 - margin, 0); // Horizontal
  line(0, -drawHeight/2 + 30, 0, drawHeight/2 - margin); // Vertical

  // Axis labels
  fill('black');
  noStroke();
  textAlign(CENTER, TOP);
  textSize(16);
  text('β₁', canvasWidth/2 - margin - 10, 10);

  textAlign(LEFT, CENTER);
  text('β₂', 10, -drawHeight/2 + 35);

  // Tick marks
  stroke(100);
  for (let i = -2; i <= 2; i++) {
    if (i !== 0) {
      // Vertical axis ticks
      line(-5, -i * scale, 5, -i * scale);
      fill('black');
      noStroke();
      textAlign(RIGHT, CENTER);
      textSize(12);
      text(i.toString(), -8, -i * scale);
      stroke(100);

      // Horizontal axis ticks
      line(i * scale, -5, i * scale, 5);
      fill('black');
      noStroke();
      textAlign(CENTER, TOP);
      text(i.toString(), i * scale, 8);
      stroke(100);
    }
  }

  pop();
}

function drawErrorContours() {
  // Draw error contour ellipses centered at OLS solution
  push();
  translate(centerX, centerY);
  noFill();

  // Multiple contours at different levels
  for (let i = 1; i <= 3; i++) {
    let rx = i * 50;
    let ry = i * 40;
    let opacity = 255 - (i - 1) * 80;
    stroke(244, 67, 54, opacity);
    strokeWeight(2 - (i - 1) * 0.4);
    ellipse(olsBeta1 * scale, -olsBeta2 * scale, rx * 2, ry * 2);
  }

  pop();
}

function drawKeyInsights(t, shrinkFactor) {
  // Draw key insights panel
  let panelX = 10;
  let panelY = drawHeight - 110;
  let panelW = 250;
  let panelH = 100;

  push();
  fill(255, 255, 255, 240);
  stroke(200);
  strokeWeight(1);
  rect(panelX, panelY, panelW, panelH, 10);

  fill('black');
  noStroke();
  textAlign(LEFT, TOP);
  textSize(12);
  text('Key Insights:', panelX + 10, panelY + 5);
  text('• L2 Penalty: Circular constraint', panelX + 10, panelY + 25);
  text('• Constraint radius: ' + t.toFixed(2), panelX + 10, panelY + 42);
  text('• Shrinkage: ' + (shrinkFactor * 100).toFixed(0) + '%', panelX + 10, panelY + 59);
  text('• Coefficients never reach zero', panelX + 10, panelY + 76);
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
    if (typeof lambdaSlider !== 'undefined') {
      lambdaSlider.size(canvasWidth - sliderLeftMargin - margin);
    }
  }
}
