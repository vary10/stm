#include "dlib/svm.h"
#include <iostream>
#include <stdlib.h> 
// #include <unordered_map>


typedef dlib::matrix<double, 2, 1> sample_type;
typedef dlib::radial_basis_kernel<sample_type> kernel_type;
typedef dlib::normalized_function<dlib::decision_function<kernel_type> > funct_type;


class stm {
public:
    funct_type learned_function;
    dlib::svm_c_trainer<kernel_type> trainer;
    dlib::vector_normalizer<sample_type> normalizer;
    // std::unordered_multimap<int, std::vector<sample_type> > basis_timer;

    stm (std::vector<sample_type> s, std::vector<double> l);

    /*!
    если константы C и gamma равны начальным значениям 1 и 0.01, 
    то set_parameters подбирает наилучшие значения по 
    максимуму суммы квадратов точности и полноты
    !*/
    void set_parameters (std::vector<sample_type>& samples, std::vector<double>& labels); // если коснтанты C и gamma равны начальным значениям 1 и 0.01, то подбирает наилучшие значения по максимуму суммы квадратов точности и полноты

    void train (std::vector<sample_type>& samples, std::vector<double>& labels);

    template <typename matrix_type>
    void denormalize (std::vector<matrix_type>& vecs);

    // добавляет новую выборку к существующим опорным векторам
    void update (std::vector<sample_type> s, std::vector<double> l);

    template <typename matrix_type>
    std::vector<sample_type> to_vector(const matrix_type& M);

    void reduce_basis (std::vector<sample_type>& samples, std::vector<double>& labels);
};