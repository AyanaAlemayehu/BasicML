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
	void backpropEnd(NeuronLayer* preceding, Eigen::VectorXd* goalVec) {

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
				wprime(j, k) = (*preceding).currentV(k) * temp * (1 - temp) * 2 * (currentV(j) - (*goalVec)(j));
			}
			double temp = sigApprox(unscaledSum(j));
			biasprime(j) = temp * (1 - temp) * 2 * (currentV(j) - (*goalVec)(j));
		}

		//now computing the partials of the previous layer (partial C_0 / partial a^(L-1)_k)
		Eigen::VectorXd partialPrev(numPrev);
		for (int k = 0; k < numPrev; k++) {
			//must sum over current layer to get this partials total value (note switch of variables)
			double sum = 0;
			for (int i = 0; i < numN; i++) {
				double temp = sigApprox(unscaledSum(i));
				sum += weights(i, k) * (temp) * (1 - temp) * 2 * (currentV(i) - (*goalVec)(i));
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

	//attempting to initialize neural network (using 3 layers cause it sucked on two)
	int layer1 = 10; //ten neurons for final layer one
	int layer2 = 16; //sixteen neurons for second to last layer
	int layer3 = 16; //sixteen neurons for third to last layer

	NeuronLayer* three = new NeuronLayer(&layer3, &size_images);
	NeuronLayer* two = new NeuronLayer(&layer2, &layer3);
	NeuronLayer* one = new NeuronLayer(&layer1, &layer2);

	int trainBatch = 128;
	int trainings = 400;
	for (int i = 0; i < trainings; i++) {
		//THIS COULD BE THE ISSUE IF THERE EXISTS ERRORS (also speed up here prolly)
		Eigen::MatrixXd weightPartialAvgLayer3(16, 784);
		weightPartialAvgLayer3 = weightPartialAvgLayer3.Zero(16, 784);
		Eigen::MatrixXd weightPartialAvgLayer2(16, 16);
		weightPartialAvgLayer2 = weightPartialAvgLayer2.Zero(16, 16);
		Eigen::MatrixXd weightPartialAvgLayer1(10, 16);//numN then numPrev
		weightPartialAvgLayer1 = weightPartialAvgLayer1.Zero(10, 16);
		Eigen::VectorXd biasPartialAvgLayer3(16);
		biasPartialAvgLayer3 = biasPartialAvgLayer3.Zero(16);
		Eigen::VectorXd biasPartialAvgLayer2(16);
		biasPartialAvgLayer2 = biasPartialAvgLayer2.Zero(16);
		Eigen::VectorXd biasPartialAvgLayer1(10);
		biasPartialAvgLayer1 = biasPartialAvgLayer1.Zero(10);


		for (int j = 0; j < trainBatch; j++) {
			//goal is to:
			//1) grab a photo
			//2) classify photo
			//3) backpropogate on photo
			//4) save weights and biases
			//step 0 is make goal vector
			Eigen::VectorXd goal(10);
			for (int k = 0; k < 10; k++) {
				goal(k) = k == (int)label_data_set[i*trainBatch + j] ? 1 : 0;//default to negative one first except if goal reading
			}
			//then on to steps 1 onwards
	
			NeuronLayer* base = new NeuronLayer(image_data_set[i*trainBatch + j], &size_images);//based upon (trainBatch*i + j)th image
			three->computeValues(base);
			two->computeValues(three);//again compute values
			one->computeValues(two);//just compute values
			//could print current error here if wanted

			//steps 1 and 2 done, now need to backprop and save weights n biases
			one->backpropEnd(two, &goal);
			two->backprop(three, &one->bpPartials);
			three->backprop(base, &two->bpPartials);

			//remember we will NEGATE these partials in the end (negative gradient)
			weightPartialAvgLayer1 = weightPartialAvgLayer1 + one->backPropWeights;
			weightPartialAvgLayer2 = weightPartialAvgLayer2 + two->backPropWeights;
			weightPartialAvgLayer3 = weightPartialAvgLayer3 + three->backPropWeights;
			biasPartialAvgLayer1 = biasPartialAvgLayer1 + one->backPropBias;
			biasPartialAvgLayer2 = biasPartialAvgLayer2 + two->backPropBias;
			biasPartialAvgLayer3 = biasPartialAvgLayer3 + three->backPropBias;
			//cout << endl << biasPartialAvgLayer1 << endl << biasPartialAvgLayer2 << endl << biasPartialAvgLayer3 << endl << "done ";
			//cout << (int)label_data_set[i * trainBatch + j] << endl;
			//cout <<goal << endl;
		}
		//now take average and subtract (stochastic method)
		one->weights = one->weights - (1.0 / trainBatch) * weightPartialAvgLayer1;
		two->weights = two->weights - (1.0 / trainBatch) * weightPartialAvgLayer2;
		three->weights = three->weights - (1.0 / trainBatch) * weightPartialAvgLayer3;
		one->biases = one->biases - (1.0 / trainBatch) * biasPartialAvgLayer1;
		two->biases = two->biases - (1.0 / trainBatch) * biasPartialAvgLayer2;
		three->biases = three->biases - (1.0 / trainBatch) * biasPartialAvgLayer3;
		cout << i << "/" << trainings << endl;
	}
	//this is effectivley training the neural network via backpropogation stochastically modeling (i think thats the word) the gradient of the cost function
	
	//TESTING NEURAL NETWORK
	while (true){
		int test_num = 0;
		cout << endl << "test NN on image: " << endl;
		cin >> test_num;
		cout << displayImageAndLabel(test_num, image_data_set, label_data_set);
	//what neural network believes
		NeuronLayer* base = new NeuronLayer(image_data_set[test_num], &size_images);//change base to reflect new number
		three->computeValues(base);//just compute values
		two->computeValues(three);//again compute values
		one->computeValues(two);
		//finally output values that neural network feels
		//values below
		cout << one->currentV;
		int max = 0;
		for (int i = 1; i < 10; i++) {
			max = one->currentV(i) > one->currentV(max) ? i : max;
		}
		int secondmax = max == 0 ? 1 : 0;
		for (int i = 0; i < 10; i++) {
			secondmax = ((one->currentV(i) > one->currentV(secondmax)) && max != i) ? i : secondmax;
		}
		cout << endl << "Best Guess: " << max << " Second Best: " << secondmax;
	}
	
}
