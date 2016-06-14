#include "stm.h"


int main() 
{
    // CREATE SAMPLES
    std::vector<sample_type> samples;
    std::vector<double> labels;
    for (float r = -10; r <= 10;)
    {
        for (float c = -10; c <= 10;)
        {
            sample_type samp;
            samp(0) = r;
            samp(1) = c;
            samples.push_back(samp);

            if (sqrt((double)(samp(0)-3)*(samp(0)-3) + samp(1)*samp(1)) <= 4 || sqrt((double)(samp(0) + 2)*(samp(0) + 2) + samp(1)*samp(1)) <= 3)
                labels.push_back(+1);
            else
                labels.push_back(-1);
            c += 0.5;
        }
        r += 0.5;
    }

    // dlib::randomize_samples(samples, labels);

    // for (int i = 0; i < samples.size(); ++i)
    // {
    //     std::cout << "[" << samples[i](0) << ", " << samples[i](1) << "], ";
    // }

    // for (int i = 0; i < labels.size(); ++i)
    // {
    //     std::cout << labels[i] << ", ";
    // }
    
    // TRAIN
    dlib::randomize_samples(samples, labels);
    stm func(samples, labels);       


    //// уменьшение базиса (
    // func.reduce_basis(samples, labels, 0.01);
    // for (size_t i = 0; i < samples.size(); ++i)
    // {
    //     samples[i] = func.normalizer(samples[i]);
    // }

    // func.learned_function.function = reduced2(func.trainer, 40).train(samples, labels);

    
    // PRINT BASIS AND LABELS
    std::vector<sample_type> basis;
    std::vector<double> label;
    basis = func.get_basis_vectors();
    std::cout << basis.size() << std::endl;
    for (int i = 0; i < basis.size(); ++i)
    {
        label.push_back(func.learned_function(basis[i]));
        std::cout << "[" << basis[i](0) << ", " << basis[i](1) << "],\n";
    }
    std::cout << label.size() << std::endl;
    func.transform(label);
    for (int i = 0; i < basis.size(); ++i)
    {
        std::cout << label[i] << ",\n";
    }

    
    // DELETE_IN_C
    // dele.resize(8);
    // dele[0] = basis[1];
    // dele[1] = basis[10];
    // dele[2] = basis[20];
    // dele[3] = basis[0];
    // dele[4] = basis[11];
    // dele[5] = basis[30];
    // dele[6] = basis[40];
    // dele[7] = basis[33];



    // DELETE
    std::vector<sample_type> dele;
    dele.resize(14);
    dele[0] = basis[35];
    dele[1] = basis[22];
    dele[2] = basis[43];
    dele[3] = basis[46];
    dele[4] = basis[70];
    dele[5] = basis[54];
    dele[6] = basis[4];
    dele[7] = basis[81];
    dele[8] = basis[27];
    dele[9] = basis[49];
    dele[10] = basis[50];
    dele[11] = basis[42];
    dele[12] = basis[13];
    dele[13] = basis[41];
    func.delete_basis(dele, false);

    
    std::vector<double> add_labels;
    std::vector<sample_type> add_samples;
    add_labels.resize(6);
    add_labels[0] = 1;
    add_labels[1] = 1;
    add_labels[2] = 1;
    add_labels[3] = 1;
    add_labels[4] = -1;
    add_labels[5] = -1;

    for (int i = 0; i < 6; ++i)
    {   
        double a, b;
        sample_type sam;
        std::cout << "new a b" << std::endl;
        std::cin >> a >> b;
        sam(0) = a;
        sam(1) = b;
        add_samples.push_back(sam);
    }
    for (int i = 0; i < 6; ++i) {
        std::cout << add_samples[i] << std::endl;
    }
    // labels += [1, 1, 1, 1, -1, -1]
    // sv += [[0, -4], [0, 4], [-0.5, -4], [-0.5, 4], [-0.5, 4.5], [-0.5, -4.5]]

    func.update(add_samples, add_labels);

    // PRINT BASIS AND LABELS

    basis = func.get_basis_vectors();
    std::cout << basis.size() << std::endl;
    std::vector<double> labelz;


    for (int i = 0; i < basis.size(); ++i)
    {
        labelz.push_back(func.learned_function(basis[i]));
        std::cout << "[" << basis[i](0) << ", " << basis[i](1) << "],\n";
    }
    std::cout << labelz.size() << std::endl;
    func.transform(labelz);
    for (int i = 0; i < basis.size(); ++i)
    {
        std::cout << labelz[i] << ",\n";
    }

    // TEST
    sample_type sample;

    sample(0) = -1;
    sample(1) = 4;
    std::cout << func.learned_function(sample) << std::endl;    

    sample(0) = -1;
    sample(1) = 2;
    std::cout << func.learned_function(sample) << std::endl;    

    sample(0) = 0;
    sample(1) = 0;
    std::cout << func.learned_function(sample) << std::endl; 

    // std::cout << "\ncross validation accuracy with only " << stm.get_basis_vectors().size() << " support vectors: " 
    //      << cross_validate_trainer(stm.trainer, samples, labels, 3) << std::endl;

    
}