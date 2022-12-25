#include "main.h"

using namespace std;

const int IMAGE_HEIGHT = 28;
const int IMAGE_WIDTH = 28;

void reverseInt(int* input) {//reverses the input
	unsigned char c1, c2, c3, c4;
	c1 = (*input >> 24) & 0xFF;
	c2 = (*input >> 16) & 0xFF;
	c3 = (*input >> 8) & 0xFF;
	c4 = (*input) & 0xFF;
	*input = ((int)c4 << 24) + ((int)c3 << 16) + ((int)c2 << 8) + c1;
	//cout << (int)c1 << " " << (int)c2 << " " << (int)c3 << " " << (int)c4 << "\n";

}
unsigned char** unpackImage(string filename) {

	ifstream file(filename, ios::binary);
	if (file.is_open()) {
		int magicnum = 0, num_rows = 0, num_cols = 0;

		file.read((char*)&magicnum, sizeof(magicnum));
		reverseInt((int*)&magicnum);//now the magic num is correct
		int num_images;
		file.read((char*)&num_images, sizeof(num_images)), reverseInt(&num_images);//grabs number of images
		file.read((char*)&num_rows, sizeof(num_rows)), reverseInt(&num_rows);
		file.read((char*)&num_cols, sizeof(num_cols)), reverseInt(&num_cols);

		int image_size = num_rows * num_cols;
		unsigned char** _dataset = new unsigned char* [num_images];
		for (int i = 0; i < num_images; i++) {
			_dataset[i] = new unsigned char[image_size];
			file.read((char*)_dataset[i], image_size);
		}
		cout << "success";
		return _dataset;
	}
	else {
		cout << "fail";
	}
}
unsigned char* unpackLabel(string filename) {

	ifstream file(filename, ios::binary);
	if (file.is_open()) {
		int magicnum = 0;

		file.read((char*)&magicnum, sizeof(magicnum));
		reverseInt((int*)&magicnum);//now the magic num is correct
		int num_labels;
		file.read((char*)&num_labels, sizeof(num_labels)), reverseInt(&num_labels);//grabs number of images

		unsigned char* _dataset = new unsigned char[num_labels];
		for (int i = 0; i < num_labels; i++) {
			file.read((char*)&_dataset[i], 1);
		}
		return _dataset;
	}
	else {
		cout << "fail";
	}
}
string displayImageAndLabel(int number, unsigned char** images, unsigned char* labels, int thresh = 127) {
	string outp = "";
	for (int j = 0; j < IMAGE_HEIGHT; j++) {
		for (int i = 0; i < IMAGE_WIDTH; i++) {
			outp += (i % IMAGE_WIDTH == 0) ? "\n" : "";
			outp += (int)images[number][i + j* IMAGE_WIDTH] < thresh ? "-" : "O";
		}
	}
	//adding label now
	outp += "\n \n";
	outp += "             " + to_string((int)labels[number]);
	outp += "\n";
	return outp;
}
double sigApprox(double inp) {
	return .5 * (((inp) / (1.0 + abs((inp))) + 1.0));
}

double square(double inp) {
	return inp * inp;
}


class NeuronLayer {
public:

	Eigen::MatrixXd weights;	//weights
	Eigen::VectorXd biases;
	Eigen::VectorXd currentV;

	//BACKPROPOGATION COMPUTATIONS
	Eigen::VectorXd unscaledSum;//basically currentV but without sigmoid applied, helpful for backpropogation calculation
	Eigen::MatrixXd backPropWeights;
	Eigen::VectorXd backPropBias;
	Eigen::VectorXd bpPartials;

	int numN;
	int numPrev;
	NeuronLayer(int* numNeurons, int* numPrecedingNeurons) {
		//weights and biases for this specific Neuron Layer	
		//dynamic eigen matrixes.

		Eigen::MatrixXd w(*numNeurons, *numPrecedingNeurons);	//weights
		Eigen::VectorXd b(*numNeurons);
		Eigen::VectorXd cv(*numNeurons);



		default_random_engine generator;
		uniform_real_distribution<double> distribution(-1.0, 1.0);
		for (int i = 0; i < *numNeurons; i++) {
			for (int j = 0; j < *numPrecedingNeurons; j++) {

				w(i, j) = distribution(generator);//randomize weights
			}
			b(i) = distribution(generator);//randomize biases
			cv(i) = 0; //default values for neurons
		}
		weights = w;
		biases = b;
		currentV = cv;
		numN = *numNeurons;
		numPrev = *numPrecedingNeurons;

	}

	NeuronLayer(unsigned char* pixels, int* numPixels) {//the initial layer constructor from pixels
		Eigen::VectorXd cv(*numPixels);
		for (int i = 0; i < *numPixels; i++) {
			cv(i) = sigApprox(pixels[i] - 127);
		}
		//note there arent weights n biases for base layer.
		currentV = cv;

	}
	void computeValues(NeuronLayer* preceding) {//computes current values from preceding layer (dont do with base layer)
		//this should do some sort of matrix multiplication (using eigen)
		Eigen::VectorXd newVals = (weights *  (*preceding).currentV + biases);
		unscaledSum = newVals;
		currentV = newVals.unaryExpr(&sigApprox);


	}

	//gets sum of squared error
	double cost(int goalReading) {//assuming that we are classifying images from 0-9. Only call if this is last layer (10 neurons)
		Eigen::VectorXd temp(10);
		for (int i = 0; i < 10; i++) {
			temp(i) = i == goalReading ? 1 : 0;//default to negative one first except if goal reading
		}
		Eigen::VectorXd diff = currentV - temp;
		diff = diff.unaryExpr(&square);//square the values
		double sum = 0;
		for (int i = 0; i < 10; i++) {
			sum += diff[i];
		}
		return sum;
	}


	//goal is to get matrix of partial derivatives responding to changes given a single training piece.
	//THIS IS THE FINAL LAYER, THUS HAS ASSUMPTIONS (like goalvec) ONLY RELEVANT IN FINAL LAYER
	void backpropEnd(NeuronLayer* preceding, Eigen::VectorXd goalVec) {

		//should go layer by layer, begining with last layer (doesnt matter tho cause I made a layer class)

		//last layer first partial cost derivative with respect to weight:
		//value of previous neuron*derivative of sigmoid approximiation(sum) *2*(current val - goal val).
		Eigen::MatrixXd wprime(numN, numPrev);
		Eigen::VectorXd biasprime(numN);

		//we will compute from top to bottom (a^(L)_0 to a^(L)_n) <- see 3blue1brown video for notation
		
		for (int j = 0; j < numN; j++) {
			for (int k = 0; k < numPrev; k++) {
				//this is partial c_0/partial w(L)_jk
				//previous value of a^(L-1)_k*sigmoid approx(sum)*2*(current val - goal val)
				//NOTE temp*(1-temp) approximates derivative of sigmoid
				double temp = sigApprox(unscaledSum(j)); 
				wprime(j, k) = (*preceding).currentV(k) * temp * (1 - temp) * 2 * (currentV(j) - goalVec(j));
			}
			double temp = sigApprox(unscaledSum(j));
			biasprime(j) = temp * (1 - temp) * 2 * (currentV(j) - goalVec(j));
		}

		//now computing the partials of the previous layer (partial C_0 / partial a^(L-1)_k)
		Eigen::VectorXd partialPrev(numPrev);
		for (int k = 0; k < numPrev; k++) {
			//must sum over current layer to get this partials total value (note switch of variables)
			double sum = 0;
			for (int i = 0; i < numN; i++) {
				double temp = sigApprox(unscaledSum(i));
				sum += weights(i, k) * (temp) * (1 - temp) * 2 * (currentV(i) - goalVec(i));
			}
			partialPrev(k) = sum;
		}
		//save all vals to be accessed by other classes
		backPropWeights = wprime;
		backPropBias = biasprime;
		bpPartials = partialPrev;
		
	}
	//THIS IS GENERALIZED BACKPROP
	void backprop(NeuronLayer* preceding, Eigen::VectorXd* partials) {

		//should go layer by layer, begining with last layer (doesnt matter tho cause I made a layer class)

		//last layer first partial cost derivative with respect to weight:
		//value of previous neuron*derivative of sigmoid approximiation(sum) *2*(current val - goal val).
		Eigen::MatrixXd wprime(numN, numPrev);
		Eigen::VectorXd biasprime(numN);


		//we will compute from top to bottom (a^(L)_0 to a^(L)_n) <- see 3blue1brown video for notation

		for (int j = 0; j < numN; j++) {
			for (int k = 0; k < numPrev; k++) {
				//this is partial c_0/partial w(L)_jk
				//previous value of a^(L-1)_k*sigmoid approx(sum)*2*(current val - goal val)
				//NOTE temp*(1-temp) approximates derivative of sigmoid


				double temp = sigApprox(unscaledSum(j));
				wprime(j, k) = (*preceding).currentV(k) * temp * (1 - temp) * (*partials)(j);
			}
			double temp = sigApprox(unscaledSum(j));
			biasprime(j) = temp * (1 - temp) * (*partials)(j);
		}

		//now computing the partials of the previous layer (partial C_0 / partial a^(L-1)_k)
		Eigen::VectorXd partialPrev(numPrev);
		for (int k = 0; k < numPrev; k++) {
			//must sum over current layer to get this partials total value (note switch of variables)
			double sum = 0;
			for (int i = 0; i < numN; i++) {
				double temp = sigApprox(unscaledSum(i));
				sum += weights(i, k) * (temp) * (1 - temp) * 2 * (*partials)(i);
			}
			partialPrev(k) = sum;
		}

		//save all vals to be accessed by other classes
		backPropWeights = wprime;
		backPropBias = biasprime;
		bpPartials = partialPrev;

	}


};

int main() {
	int num_images = 60000;//not gonna find out why sizeof doesnt work rn dont feel like it just gonna hardcode it right here
	int size_images = 784;
	int current_num = 130;
	unsigned char** image_data_set = unpackImage("data/train-images.idx3-ubyte");
	unsigned char* label_data_set = unpackLabel("data/train-labels.idx1-ubyte");
	cout << displayImageAndLabel(current_num, image_data_set, label_data_set);//displaying example 130th image in training set

	//attempting to initialize neural network
	int layer1 = 10; //ten neurons for final layer one
	int layer2 = 10; //ten neurons for second to last layer


	//GOAL VECTOR FOR EVENTUAL BACKPROPOGATION USE
	Eigen::VectorXd goal(10);
	for (int i = 0; i < 10; i++) {
		goal(i) = i == (int)label_data_set[current_num] ? 1 : 0;//default to negative one first except if goal reading
	}

	NeuronLayer* base = new NeuronLayer(image_data_set[0], &size_images);//based upon the first image
	NeuronLayer* one = new NeuronLayer(&layer1, &size_images);
	NeuronLayer* two = new NeuronLayer(&layer2, &layer1);

	//cout << base->currentV << endl;
	cout << one->currentV << endl;
	
	//test doing the matrix multiplication
	one->computeValues(base);
	cout << endl << one->currentV << endl;

	//test doing matrix mult for second layer
	cout << "SECOND LAYER" << endl;
	two->computeValues(one);
	cout << endl << two->currentV << endl;

	//compute the sum squared error
	cout << endl << two->cost((int)label_data_set[current_num]);


	//REMEMEMMMMBERRR THAT IT IS THE NEGATIVE DIFFFERENCE OF THE GRADIENT, NOT THE GRADIENT ITSELF

	//as of now I should have all the steps necessary minus the negative gradient stuff to make a neural network.

	//BACKPROPOGATION ATTEMPTS
	cout << endl << "------------------------" << endl;
	two->backpropEnd(one, goal); //BIASES LOOK GOOD WOO
	cout << endl << endl << two->backPropBias;
	cout << endl << "------layertwo-------" << endl;
	one->backprop(base, &two->bpPartials);
	cout << endl << one->backPropBias << endl;

	//i will now copy my code and make an attempt at training.
}
