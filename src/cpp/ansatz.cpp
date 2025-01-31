#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <stdio.h>
#include <cudaq/algorithms/draw.h>

#include <cmath>
#include <iostream>
#include <fstream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include <random>

struct Instruct
{
    int operation; // 0=rx, 1=ry, 2=rz, 3=cx
    int src;       // rotation parameter or source qubit
    int dest;      // destination qubit
};

std::vector<Instruct> read_ints_data(char *filename)
{
    std::vector<Instruct> v;
    std::ifstream in_file(filename);
    std::string line;

    std::cout << filename << std::endl;
    if (!in_file.is_open())
    {
        std::cerr << "Cannot open specified file." << std::endl;
        return v;
    }

    while (std::getline(in_file, line))
    {
        std::stringstream ss(line);
        Instruct instr;
        char comma;

        ss >> instr.operation >> comma >> instr.src >> comma >> instr.dest;
        v.push_back(instr);
    }

    return v;
}

std::vector<double> RandomInitPar(int size_)
{
    std::vector<double> vec(size_);

    // Create a random number generator
    std::random_device rd;                          // Seed
    std::mt19937 gen(rd());                         // Mersenne Twister engine
    std::uniform_real_distribution<> dis(0.0, 1.0); // Range [0.0, 1.0)

    // Fill the vector with random doubles
    for (auto &elem : vec)
    {
        elem = dis(gen);
    }
    return vec;
}

__qpu__ void MakeCircuit(cudaq::qview<> qc, std::vector<Instruct> instr, std::vector<double> init_params)
{
    for (const auto &instruction : instr)
    {
        if (instruction.operation == 0)
        {
            rx(init_params[instruction.src], qc[instruction.dest]);
        }
        else if (instruction.operation == 1)
        {
            ry(init_params[instruction.src], qc[instruction.dest]);
        }
        else if (instruction.operation == 2)
        {
            rz(init_params[instruction.src], qc[instruction.dest]);
        }
        else if (instruction.operation == 3)
        {
            x<cudaq::ctrl>(qc[instruction.src], qc[instruction.dest]);
        }
    }
}

struct RunMakeAnsatz
{
    __qpu__ auto operator()(int n_qubits, std::vector<Instruct> instr, std::vector<double> init_params)
    {
        cudaq::qvector qc(n_qubits);
        MakeCircuit(qc, instr, init_params);
        mz(qc);
    }
};



int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Provide an input file as an argument." << std::endl;
        return 1;
    }

    auto insts = read_ints_data(argv[1]);
    std::vector<double> init_par = RandomInitPar(60);
    auto max = *std::max_element(insts.begin(), insts.end(), [](const Instruct &a, const Instruct &b)
                                 { return a.dest < b.dest; });
    int qubit_count = max.dest + 1;

    /*for (const auto &instr : insts) {
        std::cout << "Operation: " << instr.operation << ", Src: " << instr.src
              << ", Dest: " << instr.dest << std::endl;
    }*/

    auto result = cudaq::sample(RunMakeAnsatz{}, qubit_count, insts, init_par);
    result.dump();
    std::cout << cudaq::draw("latex", RunMakeAnsatz{}, qubit_count, insts, init_par);

    return 0;
}
