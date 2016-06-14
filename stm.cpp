#include "stm.h"


stm::stm (
          std::vector<sample_type> samples,
          std::vector<double> labels
          )
{
    train(samples, labels);
}


void stm::train (
                 std::vector<sample_type>& samples,
                 std::vector<double>& labels
                 )
{
    // нормализуем и перемешиваем выборку
    normalizer.train(samples);
    std::vector<sample_type> train_data;
    train_data.resize(samples.size());
    for (size_t i = 0; i < samples.size(); ++i)
    {
        train_data[i] = normalizer(samples[i]);
    }
    // устанавливаем оптимальные константы
    if (trainer.get_kernel().gamma == 0.1 && 
        trainer.get_c_class1() == 1)
    {
        set_parameters(train_data, labels);
    }
    // обучаем
    learned_function.normalizer = normalizer;
    learned_function.function = trainer.train(train_data, labels);
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
                std::cout << ct << "===" << gamma << std::endl; 
            }
        }
    }
    // 0.03125
    // 78125
    trainer.set_kernel(kernel_type(gamma));
    trainer.set_c(C);
    std::cout << C << " <-C, gamma-> " << gamma << std::endl;
}



void stm::sgn (std::vector<double>& labels)
{
    for (size_t i = 0; i < labels.size(); ++i)
    {
        labels[i] = (labels[i] > 0) ? 1 : -1;
    }
}


template <typename matrix_type>
void stm::to_vector (const matrix_type& M, std::vector<sample_type>& vec)
{
    vec.resize(M.nr());
    for (size_t i = 0; i < M.nr(); ++i )
    {
        vec[i] = M(i);
    }
}


template <typename matrix_type>
void stm::denormalize (std::vector<matrix_type>& vecs)
{
    for (size_t j = 0; j < vecs.size(); ++j)
    {
        for (size_t i = 0; i < vecs[0].nr(); ++i)
        {
            if (normalizer.std_devs()(i, 0) != 0)
            {
                vecs[j](i, 0) = vecs[j](i, 0) / normalizer.std_devs()(i, 0) + normalizer.means()(i, 0);
            } else
            {
                vecs[j](i, 0) = normalizer.means()(i, 0);
            }
            
        }
    }
}


// void stm::reduce_basis (std::vector<sample_type>& samples, std::vector<double>& labels, double eps)
// {
//     normalizer.train(samples);
//     std::vector<sample_type> train_data;
//     train_data.resize(samples.size());
//     for (size_t i = 0; i < samples.size(); ++i)
//     {
//         train_data[i] = normalizer(samples[i]);
//     }

//     const dlib::matrix<double, 1, 2> full_acc = dlib::cross_validate_trainer(trainer, train_data, labels, 3);
//     size_t n = 40;
//     dlib::matrix<double, 1, 2> n_acc = dlib::cross_validate_trainer(reduced2(trainer, n), train_data, labels, 3);
//     std::cout << n << "=" << n_acc << "----" << full_acc << std::endl;
//     while (fabs(n_acc(0) - full_acc(0)) > eps || fabs(n_acc(1) - full_acc(1)) > eps) {
//         n += 5;
//         n_acc = dlib::cross_validate_trainer(reduced2(trainer, n), train_data, labels, 3);
//         std::cout << fabs(n_acc(0) - full_acc(0)) << " " << fabs(n_acc(1) - full_acc(1)) << " " << eps << std::endl;
//         std::cout << n << "=" << n_acc << "----" << full_acc << std::endl;
//     }  
//     if (trainer.get_kernel().gamma == 0.1 && trainer.get_c_class1() == 1)
//     {
//         set_parameters(train_data, labels);
//     }
//     std::cout << "\n\n\n" << n <<  std::endl;
//     learned_function.normalizer = normalizer;
//     learned_function.function = reduced2(trainer, n).train(train_data, labels);
// }


std::vector<sample_type> stm::get_basis_vectors ()
{
    std::vector<sample_type> basis;
    to_vector(learned_function.function.basis_vectors, basis);
    denormalize(basis);
    return basis;
}


void stm::delete_basis (std::vector<sample_type>& basis, bool upd)
{
    to_delete.reserve(to_delete.size() + basis.size());
    to_delete.insert(to_delete.end(), basis.begin(), basis.end());
    if (upd == true)
    {
        commit_delete();
        to_delete.clear();
    }
}


// добавляет новую выборку к существующим опорным векторам
void stm::update (std::vector<sample_type>& samples, std::vector<double>& labels)
{
    std::vector<double> new_labels;
    std::vector<sample_type> new_samples;
    new_samples = get_basis_vectors();
    for (size_t i = 0; i < to_delete.size(); ++i)
    {
        for (size_t j = 0; j < new_samples.size(); ++j)
        {
            if (stm::vec_eq(to_delete[i], new_samples[j]))
            {
                new_samples.erase(new_samples.begin() + j);
                break;
            }
        }
    }
    to_delete.clear();
    new_labels.resize(new_samples.size());
    for (size_t i = 0; i < new_samples.size(); ++i)
    {
        new_labels[i] = learned_function(new_samples[i]);
    }
    sgn(new_labels);
    new_samples.reserve(new_samples.size() + samples.size());
    new_samples.insert(new_samples.end(), samples.begin(), samples.end());
    new_labels.reserve(new_labels.size() + labels.size());
    new_labels.insert(new_labels.end(), labels.begin(), labels.end());
    train(new_samples, new_labels);
}


void stm::commit_delete ()
{
    std::vector<double> new_labels;
    std::vector<sample_type> new_samples;
    new_samples = get_basis_vectors();
    for (size_t i = 0; i < to_delete.size(); ++i)
    {
        for (size_t j = 0; j < new_samples.size(); ++j)
        {
            if (stm::vec_eq(to_delete[i], new_samples[j]))
            {
                std::cout << "DELETING" << new_samples[j] << std::endl;
                new_samples.erase(new_samples.begin() + j);
                break;
            }
        }
    }

    std::cout << get_basis_vectors().size() << " != " << new_samples.size() << std::endl; 

    new_labels.resize(new_samples.size());
    for (size_t i = 0; i < new_samples.size(); ++i)
    {
        std::cout << "[" << new_samples[i](0) << ", " << new_samples[i](1) << "] -- " << learned_function(new_samples[i])<< std::endl;
        new_labels[i] = learned_function(new_samples[i]);
    }
    sgn(new_labels);
    normalizer.train(new_samples);
    learned_function.normalizer = normalizer;

    train(new_samples, new_labels);
}


bool stm::vec_eq (sample_type& a, sample_type& b)
{
    double eps = 0;
    for (size_t i = 0; i < a.nr(); ++i)
    {
        eps += fabs(a(i) - b(i));
    }
    return (eps < 0.01 * a.nr());
}

