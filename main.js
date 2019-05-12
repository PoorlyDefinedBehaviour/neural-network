class NeuralNetwork {
  constructor() {
    this.w1 = Math.random() * 0.2 - 0.1;
    this.w2 = Math.random() * 0.2 - 0.1;
    this.b = Math.random() * 0.2 - 0.1;
    this.learning_rate = 0.1;
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  predict(input) {
    return this.sigmoid(input[0] * this.w1 + input[1] * this.w2 + this.b);
  }

  train(data, iterations) {
    for (let i = 0; i < iterations; ++i) {
      const point = data[Math.floor(Math.random() * (data.length - 1))];
      const target = point[2];

      const z = this.w1 * point[0] + this.w2 * point[1] + this.b;
      const prediction = this.sigmoid(z);

      const cost = (prediction - target) ** 2;
      const d_cost = 2 * (prediction - target);

      const d_prediction = this.sigmoid(z) * (1 - this.sigmoid(z));

      const dz_dw1 = point[0];
      const dz_dw2 = point[1];
      const dz_db = 1;

      const dcost_dw1 = d_cost * d_prediction * dz_dw1;
      const dcost_dw2 = d_cost * d_prediction * dz_dw2;
      const dcost_db = d_cost * d_prediction * dz_db;

      this.w1 -= this.learning_rate * dcost_dw1;
      this.w2 -= this.learning_rate * dcost_dw2;
      this.b -= this.learning_rate * dcost_db;
    }
  }
}

/* training set [length, width, color(0=blue and 1=red)] */
const data = [
  [2, 1, 0],
  [3, 1, 0],
  [2, 0.5, 0],
  [1, 1, 0],
  [3, 1.5, 1],
  [3.5, 0.5, 1],
  [4, 1.5, 1],
  [5.5, 1, 1],
];

const unlabeled_data = [4.5, 1, 'should be 1'];

const nn = new NeuralNetwork();
nn.train(data, 50000);
console.log(
  'unlabeled_data ->',
  unlabeled_data,
  'prediction ->',
  nn.predict(unlabeled_data)
);
for (const input of data)
  console.log('input ->', input, 'prediction ->', nn.predict(input));
