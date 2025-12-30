// Lasso Regression Geometry MicroSim
// Visualizes L1 regularization constraint and Lasso solution with feature selection

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
let lambda = 0.5; // Regularization strength
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

  describe('Interactive visualization of Lasso regression L1 regularization geometry showing constraint diamond and sparsity-inducing solution', LABEL);
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
  text('Lasso Regression Geometry (L1 Regularization)', canvasWidth / 2, 10);

  // Reset text settings
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);

  // Draw coordinate axes
  drawAxes();

  // Calculate L1 constraint size from lambda
  // As lambda increases, constraint gets tighter (smaller diamond)
  let constraintSize = scale * (1.2 - lambda * 0.8); // size in pixels

  // Draw L1 constraint diamond
  push();
  translate(centerX, centerY);
  fill(76, 175, 80, 50); // Green with transparency
  stroke(76, 175, 80);
  strokeWeight(3);

  // Diamond: |β₁| + |β₂| ≤ t
  beginShape();
  vertex(constraintSize, 0);  // Right
  vertex(0, -constraintSize);  // Top
  vertex(-constraintSize, 0);  // Left
  vertex(0, constraintSize);   // Bottom
  endShape(CLOSE);

  // Label the constraint
  fill(76, 175, 80);
  noStroke();
  textAlign(CENTER);
  textSize(14);
  text('|β₁| + |β₂| ≤ t', 0, constraintSize + 20);
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

  // Calculate Lasso solution (often hits corner)
  // Simplified: Project OLS toward nearest corner of diamond
  let lassoBeta1, lassoBeta2;
  let t = constraintSize / scale;

  // Check if OLS is inside constraint
  if (abs(olsBeta1) + abs(olsBeta2) <= t) {
    // OLS is inside, use OLS
    lassoBeta1 = olsBeta1;
    lassoBeta2 = olsBeta2;
  } else {
    // Simplified Lasso solution: tends toward corners
    // For high lambda, solution is on axis (one coefficient = 0)
    if (lambda > 0.3) {
      // Solution on β₁ axis (β₂ = 0) - feature selection!
      lassoBeta1 = min(olsBeta1, t);
      lassoBeta2 = 0;
    } else {
      // Solution on edge of diamond
      let ratio1 = abs(olsBeta1) / (abs(olsBeta1) + abs(olsBeta2));
      lassoBeta1 = sign(olsBeta1) * ratio1 * t;
      lassoBeta2 = sign(olsBeta2) * (1 - ratio1) * t;
    }
  }

  // Draw Lasso solution
  push();
  translate(centerX, centerY);
  fill(76, 175, 80); // Green
  noStroke();
  circle(lassoBeta1 * scale, -lassoBeta2 * scale, 14);
  fill(76, 175, 80);
  textAlign(LEFT);
  textSize(14);
  text('Lasso', lassoBeta1 * scale + 10, -lassoBeta2 * scale - 5);

  // If on axis, show feature selection
  if (abs(lassoBeta2) < 0.01) {
    textSize(12);
    fill(255, 152, 0); // Orange
    text('(β₂ = 0)', lassoBeta1 * scale + 10, -lassoBeta2 * scale + 12);
  }
  pop();

  // Draw movement to corner
  if (abs(olsBeta1 - lassoBeta1) > 0.01 || abs(olsBeta2 - lassoBeta2) > 0.01) {
    push();
    translate(centerX, centerY);
    stroke(100);
    strokeWeight(2);
    drawingContext.setLineDash([5, 5]);
    line(olsBeta1 * scale, -olsBeta2 * scale, lassoBeta1 * scale, -lassoBeta2 * scale);
    drawingContext.setLineDash([]);

    // Arrow head
    let angle = atan2(-(lassoBeta2 - olsBeta2), lassoBeta1 - olsBeta1);
    let arrowSize = 10;
    push();
    translate(lassoBeta1 * scale, -lassoBeta2 * scale);
    rotate(angle);
    fill(100);
    noStroke();
    triangle(0, 0, -arrowSize, -arrowSize/2, -arrowSize, arrowSize/2);
    pop();
    pop();
  }

  // Highlight corner if solution is there
  if (abs(lassoBeta2) < 0.01 && lambda > 0.3) {
    push();
    translate(centerX, centerY);
    noFill();
    stroke(255, 152, 0);
    strokeWeight(2);
    drawingContext.setLineDash([3, 3]);
    circle(lassoBeta1 * scale, 0, 30);
    drawingContext.setLineDash([]);

    fill(255, 152, 0);
    noStroke();
    textAlign(CENTER);
    textSize(11);
    text('Feature\nSelection!', lassoBeta1 * scale, 55);
    pop();
  }

  // Draw key insights panel
  drawKeyInsights(t, lassoBeta2);

  // Control labels
  fill('black');
  noStroke();
  textAlign(LEFT, CENTER);
  textSize(defaultTextSize);
  text('Regularization λ: ' + lambda.toFixed(2), 10, drawHeight + 20);
}

function sign(x) {
  return x >= 0 ? 1 : -1;
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

function drawKeyInsights(t, lassoBeta2) {
  // Draw key insights panel
  let panelX = 10;
  let panelY = drawHeight - 110;
  let panelW = 260;
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
  text('• L1 Penalty: Diamond constraint', panelX + 10, panelY + 25);
  text('• Constraint size: ' + t.toFixed(2), panelX + 10, panelY + 42);

  if (abs(lassoBeta2) < 0.01) {
    fill(255, 152, 0);
    text('• Sparsity: β₂ = 0 (FEATURE SELECTION)', panelX + 10, panelY + 59);
  } else {
    fill('black');
    text('• Diamond corners promote sparsity', panelX + 10, panelY + 59);
  }

  fill('black');
  text('• Sharp corners → exact zeros', panelX + 10, panelY + 76);
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
