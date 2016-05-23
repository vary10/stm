#include "stm.h"


stm::stm (std::vector<sample_type> s, std::vector<double> l)
{
    train(s, l);
}


void stm::train (std::vector<sample_type>& samples, std::vector<double>& labels)
{
    // нормализуем и перемешиваем выборку
    normalizer.train(samples);
    for (unsigned long i = 0; i < samples.size(); ++i)
    {
        samples[i] = normalizer(samples[i]);
    }
    dlib::randomize_samples(samples, labels);
    // устанавливаем оптимальные константы
    if (trainer.get_kernel().gamma == 0.1 and trainer.get_c_class1() == 1)
    {
        set_parameters(samples, labels);
    }
    // обучаем
    learned_function.normalizer = normalizer;
    learned_function.function = trainer.train(samples, labels);
}


void stm::set_parameters (std::vector<sample_type>& samples, std::vector<double>& labels)
{
    double gamma;
    double C;
    double max_acc = 0;
    for (double g = 0.00001; g <= 1; g *= 5)
    {
        for (double ct = 1; ct < 100000; ct *= 5)
        {
            trainer.set_kernel(kernel_type(g));
            trainer.set_c(ct);
            const dlib::matrix<double, 1, 2> tmp = dlib::cross_validate_trainer(trainer, samples, labels, 3);
            if (max_acc < tmp(0) * tmp(0) + tmp(1) * tmp(1))
            {
                max_acc = tmp(0) * tmp(0) + tmp(1) * tmp(1);
                gamma = g;
                C = ct;
            }
        }
    }
    trainer.set_kernel(kernel_type(gamma));
    trainer.set_c(C);
}


// добавляет новую выборку к существующим опорным векторам
void stm::update (std::vector<sample_type> s, std::vector<double> l)
{
    std::vector<sample_type> tmp = to_vector(learned_function.function.basis_vectors);
    denormalize(tmp);
    s.insert(s.end(), tmp.begin(), tmp.end());
    // std::vector<double> tmp_labels = learned_function.get_labels();
    // l.insert(l.end(), tmp_labels.begin(), tmp_labels.end());
    // train(s, l);
}




template <typename matrix_type>
std::vector<sample_type> stm::to_vector (const matrix_type& M)
{
    std::vector<sample_type>* vecs = new std::vector<sample_type>;
    for (unsigned long i = 0; i < M.nr(); ++i )
    {
        (*vecs).push_back(M(i));
    }
    return *vecs;
}


template <typename matrix_type>
void stm::denormalize (std::vector<matrix_type>& vecs)
{
    for (unsigned long j = 0; j < vecs.size(); ++j)
    {
        for (unsigned long i = 0; i < vecs[0].nr(); ++i)
        {
            vecs[j](i, 0) = (normalizer.std_devs()(i, 0) != 0) ?
                          vecs[j](i, 0) / normalizer.std_devs()(i, 0) + normalizer.means()(i, 0) :
                          normalizer.means()(i, 0);
        }
    }
}


void stm::reduce_basis (std::vector<sample_type>& samples, std::vector<double>& labels)
{
    const dlib::matrix<double, 1, 2> full_acc = dlib::cross_validate_trainer(trainer, samples, labels, 3);
    unsigned long n = 10;
    dlib::matrix<double, 1, 2> n_acc = dlib::cross_validate_trainer(reduced2(trainer, n), samples, labels, 3);
    while (fabs(n_acc(0) - full_acc(0)) > 0.005 and fabs(n_acc(1) - full_acc(1)) > 0.005) {
        ++n;
    }
    learned_function.function = reduced2(trainer, n).train(samples, labels);
}