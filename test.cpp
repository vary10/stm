#include "stm.h"


int main() 
{
    std::vector<sample_type> samples;
    std::vector<double> labels;
    for (int r = -20; r <= 20; ++r)
    {
        for (int c = -20; c <= 20; ++c)
        {
            sample_type samp;
            samp(0) = r;
            samp(1) = c;
            samples.push_back(samp);

            if (sqrt((double)r*r + c*c) <= 10)
                labels.push_back(+1);
            else
                labels.push_back(-1);
        }
    }

    stm stm(samples, labels);

    sample_type sample;

    sample(0) = 3.123;
    sample(1) = 2;
    std::cout << "This is a +1 class example, the classifier output is " << stm.learned_function(sample) << std::endl;

    sample(0) = 3.123;
    sample(1) = 9.3545;
    std::cout << "This is a +1 class example, the classifier output is " << stm.learned_function(sample) << std::endl;

    sample(0) = 13.123;
    sample(1) = 9.3545;
    std::cout << "This is a -1 class example, the classifier output is " << stm.learned_function(sample) << std::endl;

    sample(0) = 13.123;
    sample(1) = 0;
    std::cout << "This is a -1 class example, the classifier output is " << stm.learned_function(sample) << std::endl;

    std::cout << "cross validation accuracy before updating with new samples: " 
         << cross_validate_trainer(stm.trainer, samples, labels, 3);

    std::vector<sample_type> new_samples;
    std::vector<double> new_labels;
    for (double r = -4.5; r <= 4.5; ++r)
    {
        for (double c = -12.4; c <= 12.4; ++c)
        {
            sample_type samp;
            samp(0) = r;
            samp(1) = c;
            new_samples.push_back(samp);

            if (sqrt((double)r*r + c*c) <= 10)
                new_labels.push_back(+1);
            else
                new_labels.push_back(-1);
        }
    }
    stm.update(new_samples, new_labels);

    std::cout << "cross validation accuracy after updating with new samples: " 
         << cross_validate_trainer(stm.trainer, samples, labels, 3) << std::endl;

    std::cout << "cross validation accuracy with all " << stm.learned_function.function.basis_vectors.nr() << " of the original support vectors: " 
         << cross_validate_trainer(stm.trainer, samples, labels, 3) << std::endl;

    stm.reduce_basis(samples, labels);

    std::cout << "\ncross validation accuracy with only " << stm.learned_function.function.basis_vectors.nr() << " support vectors: " 
         << cross_validate_trainer(stm.trainer, samples, labels, 3) << std::endl;
}