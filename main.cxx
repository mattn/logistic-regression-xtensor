#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:26451 6262 26812 6297 4244)
#endif

#include <xtensor/xview.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor-blas/xlinalg.hpp>

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <fstream>

static float
softmax(xt::xarray<float> w, xt::xarray<float> x) {
  auto v = xt::linalg::dot(w, x)[0];
  return 1.0f / (1.0f + std::exp(-v));
}

static float
predict(xt::xarray<float> w, xt::xarray<float> x) {
  return softmax(w, x);
}

static xt::xarray<float>
logistic_regression(xt::xarray<float> X, xt::xarray<float> y, float rate, int ntrains) {
  xt::xtensor<float, 1> w = xt::random::randn<float>({X.shape(1)});

  for (auto n = 0; n < ntrains; n++) {
    for (auto i = 0; i < X.shape(0); i++) {
      auto x = xt::row(X, i);
      auto pred = softmax(x, w);
      auto perr = xt::row(y, i)[0] - pred;
      auto scale = rate * perr * pred * (1 - pred);
      auto dx = x * scale;
      for (auto j = 0; j < X.shape(1); j++) {
        w += dx;
      }
    }
  }

  return w;
}

static std::vector<std::string>
split(std::string& fname, char delimiter) {
  std::istringstream f(fname);
  std::string field;
  std::vector<std::string> result;
  while (getline(f, field, delimiter)) {
    result.push_back(field);
  }
  return result;
}

int main() {
  xt::random::seed(time(NULL));
  std::ifstream ifs("iris.csv");

  std::string line;

  // skip header
  std::getline(ifs, line);

  std::vector<float> rows;
  std::vector<std::string> names;
  while (std::getline(ifs, line)) {
    // sepal length, sepal width, petal length, petal width, name
    auto cells = split(line, ',');
    rows.push_back(std::stof(cells.at(0)));
    rows.push_back(std::stof(cells.at(1)));
    rows.push_back(std::stof(cells.at(2)));
    rows.push_back(std::stof(cells.at(3)));
    names.push_back(cells.at(4));
  }
  // make vector 4 dimentioned
  auto X = xt::adapt(rows, {(std::size_t)rows.size()/4, (std::size_t)4});

  // make onehot values of names
  std::map<std::string, size_t> labels;
  for(auto& name : names) {
    if (labels.count(name) == 0) labels[name] = labels.size();
  }
  std::vector<float> counts;
  for (auto& name : names) {
    if (labels.count(name) > 0) counts.push_back((float)labels[name]);
  }
  auto y = xt::adapt(counts, {(std::size_t) 1, counts.size()});
  y /= (float) labels.size();

  names.clear();
  for(auto& k : labels) {
    names.push_back(k.first);
  }

  // make factor from input values
  auto w = logistic_regression(X, y, 0.1f, 5000);
  std::cout << w << std::endl;

  // predict samples
  for (auto i = 0; i < X.shape(0); i++) {
    auto x = xt::row(X, i);
    auto n = (size_t) ((double) predict(w, x) * (double) labels.size());
    //std::cout << n << " " << predict(w, x) << std::endl;
    if (n > names.size() - 1) n = names.size() - 1;
    std::cout << names[n] << std::endl;
  }

  return 0;
}
