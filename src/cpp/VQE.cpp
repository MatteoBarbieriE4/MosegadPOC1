#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <ostream>
#include <random>
#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <stdio.h>
#include <cudaq/algorithms/draw.h>

std::vector<std::string> GateList(std::string name)
{
    std::ifstream file(name);
    std::string line;
    std::map<std::string, double> data;

    // Read the file content
    if (file.is_open())
    {
        std::getline(file, line);
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file";
        return {"1", "1"};
    }

    // Remove the curly braces
    line = line.substr(1, line.size() - 2);

    // Parse the content
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        std::string key = item.substr(1, item.find(':') - 2);
        double value = std::stod(item.substr(item.find(':') + 1));
        // Remove single quotes from the key
        key.erase(remove(key.begin(), key.end(), '\''), key.end());
        data[key] = value;
    }

    // Store keys and values in vectors
    std::vector<std::string> keys;
    std::vector<double> values;
    for (const auto &pair : data)
    {
        keys.push_back(pair.first);
        values.push_back(pair.second);
    }

    return keys;
}

std::vector<double> CoeffList(std::string name)
{
    std::ifstream file(name);
    std::string line;
    std::map<std::string, double> data;

    // Read the file content
    if (file.is_open())
    {
        std::getline(file, line);
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file";
        return {1, 1};
    }

    // Remove the curly braces
    line = line.substr(1, line.size() - 2);

    // Parse the content
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        std::string key = item.substr(1, item.find(':') - 2);
        double value = std::stod(item.substr(item.find(':') + 1));
        // Remove single quotes from the key
        key.erase(remove(key.begin(), key.end(), '\''), key.end());
        data[key] = value;
    }

    // Store keys and values in vectors
    std::vector<std::string> keys;
    std::vector<double> values;
    for (const auto &pair : data)
    {
        keys.push_back(pair.first);
        values.push_back(pair.second);
    }

    return values;
}

struct Instruct
{
    int operation; // 0=rx, 1=ry, 2=rz, 3=cx
    int src;       // rotation parameter or source qubit
    int dest;      // destination qubit
};

std::vector<Instruct> read_ints_data(std::string filename)
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

__qpu__ void MakeAnsatz(cudaq::qview<> qc, std::vector<Instruct> instr, std::vector<double> init_params)
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
        MakeAnsatz(qc, instr, init_params);
        mz(qc);
    }
};

cudaq::spin_op MakeHamiltonian(std::vector<std::string> gates, std::vector<double> coeff)
{
    std::vector<char> g = {'I', 'X', 'Y', 'Z'};
    cudaq::spin_op ham = 0;
    for (int i = 0; i < gates.size(); i++)
    {
        cudaq::spin_op term = 0;
        term *= coeff[i];
        for (int j = 0; j < gates[i].size(); j++)
        {
            if (gates[i][j] == g[0])
            {
                term *= cudaq::spin::i(j);
            }
            else if (gates[i][j] == g[1])
            {
                term *= cudaq::spin::x(j);
            }
            else if (gates[i][j] == g[2])
            {
                term *= cudaq::spin::y(j);
            }
            else if (gates[i][j] == g[3])
            {
                term *= cudaq::spin::z(j);
            }
        }
        //term.dump();
        ham += term;
    }
    return ham;
}

struct VQE_step
{
    __qpu__ auto operator()(int n_qubits, std::vector<Instruct> instr, std::vector<double> init_params)
    {
        cudaq::qvector qc(n_qubits);
        MakeAnsatz(qc, instr, init_params);
    }
};

std::vector<double> RandParArray(int list_size){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 2 * M_PI);
    std::vector<double> par(list_size);

    for (int i = 0; i < list_size; ++i) {
        par[i] = dis(gen);
    }
    return par;
}


double RunSpinHam(std::string data_folder)
{
    std::vector<std::string> keys = GateList(data_folder + "/spin_ham.txt");
    std::vector<double> values = CoeffList(data_folder + "/spin_ham.txt");
    //std::cout << values[0];
    std::vector<Instruct> instr = read_ints_data(data_folder +"/spin_ops.txt");
    cudaq::spin_op H = MakeHamiltonian(keys, values);
    H.dump();
    int n_qubits = 8;
    int depth = 20;
    int list_size = n_qubits + n_qubits*depth;
    std::vector<double> rand_par = RandParArray(list_size);
    std::vector<double> quaq_par = {1.13441375, -0.78144957,  2.28057738,  2.20203496,  0.88010947,
        0.90828182, -1.4845915 ,  0.93585999,  0.72528133,  3.08889231,
        2.503021  , -1.11615998, -2.05395423, -1.76639041, -0.85913057,
       -1.90947885, -0.38830028, -0.07211347,  3.03649015, -0.69039433,
       -3.13125519,  0.44246386, -0.74408233, -2.58531782,  1.85776623,
        1.68202914, -1.19699484,  2.56786684, -0.71118619,  1.94691726,
       -0.52528608,  1.58473959, -2.82528222, -0.53588203,  2.04974605,
       -2.87251446, -2.25735681,  1.24915613,  1.29871078, -1.72821635,
       -2.37863193, -0.44081896,  2.91324575, -0.20784128, -0.08982667,
       -0.03266221,  0.05106199,  2.19335784,  0.8338011 ,  2.03776251,
        3.05932269, -0.73467854, -1.22685246,  0.78184028, -0.42230843,
        1.07274227,  0.56295417, -1.61555047,  1.24141705, -2.57815929,
        2.62724286,  1.98603243,  0.32271338, -2.47900282, -1.2008525 ,
       -3.00854019, -3.09899135,  2.95136909,  3.06970542, -0.02381676,
       -2.15746956, -2.33398825,  2.64044405,  0.23095703, -2.4655746 ,
       -0.88281894,  1.51633708, -2.83288074, -2.32053835, -1.14996385,
       -2.05139168, -1.65504237, -0.7176756 , -1.12449286, -1.89473513,
        2.22076741, -0.89677294, -1.23540521, -2.90937404, -2.78954722,
        1.53669856, -0.52159177, -1.55638185, -0.11821143,  0.89791595,
       -2.8507729 , -1.92087137,  2.4455163 , -0.39081604,  2.82520971,
       -2.15509727,  3.04756175, -0.49833032,  1.26337804, -2.45019838,
       -0.55590672,  1.15610786,  0.68111116, -1.16260409, -1.30245138,
        2.57320927,  2.89956612, -1.59550717,  0.52605506, -1.62646918,
       -2.97848402,  0.71562762,  0.56101057,  0.51626313, -2.83840812,
       -1.73758686,  0.99476873, -1.75340052, -0.4644092 ,  0.34995026,
        1.76656799, -1.05896264,  2.02109457, -1.21074207, -0.53501401,
        2.07616996,  0.11021694,  2.95820029, -0.66722439, -2.98626666,
       -2.97799518,  1.98469984, -0.18898151,  0.10771241, -2.85523342,
        1.14536594,  1.61820773, -0.69134616, -0.8844741 ,  1.46183941,
       -0.30671248, -2.44642831,  1.19209043, -2.65657168,  2.88676312,
       -0.72307121,  0.96508006,  1.23636514, -2.27049843, -1.69329163,
        3.10033258, -0.40091799,  1.84337301,  2.67214161, -1.02714399,
        1.31045657,  0.45123008,  1.67952399, -1.19889355,  1.09851728,
       -2.64695876,  1.37100103,  0.91924876};
    std::vector<double> range(list_size);
    for (int i=0; i < list_size; i++) {
        range[i] = double(i);
    }
    std::cout << (list_size == quaq_par.size()) << std::endl;
    double energy = cudaq::observe(VQE_step{},H,n_qubits,instr,quaq_par).expectation();
    std::cout << cudaq::draw("latex", VQE_step{},n_qubits,instr,range) << std::endl;
    return energy;
}

double RunN2Ham(std::string data_folder)
{
    std::vector<std::string> keys = GateList(data_folder + "/N2_ham.txt");
    std::vector<double> values = CoeffList(data_folder + "/N2_ham.txt");
    std::vector<Instruct> instr = read_ints_data(data_folder + "/N2_ops.txt");
    cudaq::spin_op H = MakeHamiltonian(keys, values);
    int n_qubits = 20;
    int depth = 10;
    int list_size = n_qubits + n_qubits*depth;
    std::vector<double> rand_par = RandParArray(list_size);
    double energy = cudaq::observe(VQE_step{},H,n_qubits,instr,rand_par);
    return energy;
}

double RunNH3Ham(std::string data_folder)
{
    std::vector<double> quaq_par = {1.13441375, -0.78144957,  2.28057738,  2.20203496,  0.88010947,
        0.90828182, -1.4845915 ,  0.93585999,  0.72528133,  3.08889231,
        2.503021  , -1.11615998, -2.05395423, -1.76639041, -0.85913057,
       -1.90947885, -0.38830028, -0.07211347,  3.03649015, -0.69039433,
       -3.13125519,  0.44246386, -0.74408233, -2.58531782,  1.85776623,
        1.68202914, -1.19699484,  2.56786684, -0.71118619,  1.94691726,
       -0.52528608,  1.58473959, -2.82528222, -0.53588203,  2.04974605,
       -2.87251446, -2.25735681,  1.24915613,  1.29871078, -1.72821635,
       -2.37863193, -0.44081896,  2.91324575, -0.20784128, -0.08982667,
       -0.03266221,  0.05106199,  2.19335784,  0.8338011 ,  2.03776251,
        3.05932269, -0.73467854, -1.22685246,  0.78184028, -0.42230843,
        1.07274227,  0.56295417, -1.61555047,  1.24141705, -2.57815929,
        2.62724286,  1.98603243,  0.32271338, -2.47900282, -1.2008525 ,
       -3.00854019, -3.09899135,  2.95136909,  3.06970542, -0.02381676,
       -2.15746956, -2.33398825,  2.64044405,  0.23095703, -2.4655746 ,
       -0.88281894,  1.51633708, -2.83288074, -2.32053835, -1.14996385,
       -2.05139168, -1.65504237, -0.7176756 , -1.12449286, -1.89473513,
        2.22076741, -0.89677294, -1.23540521, -2.90937404, -2.78954722,
        1.53669856, -0.52159177, -1.55638185, -0.11821143,  0.89791595,
       -2.8507729 , -1.92087137,  2.4455163 , -0.39081604,  2.82520971,
       -2.15509727,  3.04756175, -0.49833032,  1.26337804, -2.45019838,
       -0.55590672,  1.15610786,  0.68111116, -1.16260409, -1.30245138,
        2.57320927,  2.89956612, -1.59550717,  0.52605506, -1.62646918,
       -2.97848402,  0.71562762,  0.56101057,  0.51626313, -2.83840812,
       -1.73758686,  0.99476873, -1.75340052, -0.4644092 ,  0.34995026,
        1.76656799, -1.05896264,  2.02109457, -1.21074207, -0.53501401,
        2.07616996,  0.11021694,  2.95820029, -0.66722439, -2.98626666,
       -2.97799518,  1.98469984, -0.18898151,  0.10771241, -2.85523342,
        1.14536594,  1.61820773, -0.69134616, -0.8844741 ,  1.46183941,
       -0.30671248, -2.44642831,  1.19209043, -2.65657168,  2.88676312,
       -0.72307121,  0.96508006,  1.23636514, -2.27049843, -1.69329163,
        3.10033258, -0.40091799,  1.84337301,  2.67214161, -1.02714399,
        1.31045657,  0.45123008,  1.67952399, -1.19889355,  1.09851728,
       -2.64695876,  1.37100103,  0.91924876};
    std::vector<std::string> keys = GateList(data_folder + "/NH3_ham.txt");
    std::vector<double> values = CoeffList(data_folder + "/NH3_ham.txt");
    std::vector<Instruct> instr = read_ints_data(data_folder + "/NH3_ops.txt");
    cudaq::spin_op H = MakeHamiltonian(keys, values);
    int n_qubits = 14;
    int depth = 6;
    int list_size = n_qubits + n_qubits*depth;
    std::vector<double> rand_par = RandParArray(list_size);
    double energy = cudaq::observe(VQE_step{},H,n_qubits,instr,rand_par);
    return energy;
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Provide a data folder as an argument." << std::endl;
        return 1;
    }
    /*std::vector<std::string> keys = GateList("spin_ham.txt");
    std::vector<double> values = CoeffList("spin_ham.txt");
    // Output the vectors to verify
    std::cout << "Keys: ";
    for (const auto &k : keys)
    {
        std::cout << k << " ";
    }
    std::cout << std::endl;

    std::cout << "Values: ";
    for (const auto &v : values)
    {
        std::cout << v << " ";
    }
    std::cout << std::endl;*/

    double energy = RunSpinHam(argv[1]);
    std::cout << energy << std::endl;

    return 0;

   
}
