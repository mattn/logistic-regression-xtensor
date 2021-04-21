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
      auto perr = y[i] - pred;
      auto scale = rate * perr * pred * (1 - pred);
      w += x * scale * x.shape(0);
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
  std::map<std::string, std::size_t> labels;
  std::vector<float> counts;
  std::vector<std::string> tmp;
  for (auto &name : names) {
    std::size_t &label = labels[name];
    if (label == 0) {
      label = labels.size();
      tmp.push_back(name);
    }
    counts.push_back((float) (label - 1));
  }
  names = tmp;

  auto y = xt::adapt(counts, {counts.size()});
  y /= (float) labels.size();

  // make factor from input values
  auto w = logistic_regression(X, y, 0.01f, 1000);

  // predict samples
  for (auto i = 0; i < X.shape(0); i++) {
    auto x = xt::row(X, i);
    auto n = (std::size_t) ((double) predict(w, x) * (double) labels.size());
    if (n > names.size() - 1) n = names.size() - 1;
    std::cout << names[n] << std::endl;
  }

  return 0;
}
