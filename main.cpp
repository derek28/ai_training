/*
 * main.cpp
 */

#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include "card.h"
#include "deck.h"
#include "pokerhand.h"
#include "strength.h"



using namespace std;

int conv(Card c) {
	return (13 * (c.GetSuit() - 1) + c.GetRank() - 1);
}

void generate(int nHand, string filename) {
	int i, j;
	ofstream file;
	file.open(filename);
	
	for (i = 0; i < nHand; i++) {
		Deck deck;
		vector <Card> hole_cards;
		vector <Card> board;
		int format[104] = {0};
		float strength = 0.0;

		// Progress update
		if (i % (nHand/100) == 0) {
			cout << 100 * i / (float)nHand << "\% completed\r";
			cout << flush;
		}

		deck.Shuffle();
		hole_cards.push_back(deck.Deal());
		hole_cards.push_back(deck.Deal());
		board.push_back(deck.Deal());
		board.push_back(deck.Deal());
		board.push_back(deck.Deal());
	//	strength = GetImmediateStrength(hole_cards, board, NULL);
		strength = GetEffectiveStrength(hole_cards, board, NULL);
//		cout << hole_cards[0] << hole_cards[1] << " ";
//		cout << board[0] << board[1] << board[2] << endl;
//		cout << "IHS = " << strength << endl;
		format[conv(hole_cards[0])] = 1;
		format[conv(hole_cards[1])] = 1;
		format[conv(board[0]) + 52] = 1;
		format[conv(board[1]) + 52] = 1;
		format[conv(board[2]) + 52] = 1;
		for (j = 0; j < 104; j++) {
			file << format[j] << ",";
		}
		file << strength << endl;
	}
	file.close();
	cout << "100\% completed." << endl;
}

// argv[1] = number of samples
// argv[2] = output file name
int main(int argc, char **argv) {
	clock_t start;
	double duration;
	int nSamp;

	srand(time(NULL));
	
	if (argc != 3) {
		cout << "Usage: ./poker <numSamples> <filename>" << endl;
		return -1;
	}

	try {
		nSamp = stoi(argv[1]);
	} 
	catch (invalid_argument const &e) {
		cerr << "Bad input: std::invalid_argument thrown." << endl;
		return -1;
	}
	catch (out_of_range const &e) {
		cerr << "Integer overflow: std::out_of_range thrown." << endl;
		return -1;
	}

	if (nSamp <= 0) {
		cerr << "Negative number. Program terminated." << endl;
		return -1;
	}

	string filename(argv[2]);
	start = clock();

	cout << "Generating data samples..." << endl;
	generate(nSamp, filename);

	duration = (clock() - start) / (double) CLOCKS_PER_SEC;
	cout << endl << "Time elapsed:" << duration << endl;
	return 0;
}
