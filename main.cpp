#include <iostream>
#include <string>
#include <thread>
#include "runner.h"


using namespace std;



void produce() {
    //here we need to read all the data files 
    //and produce a frame

    
}

int main() {

    cout << "enter run, bench, or verify"  << endl;

    string input;
    cin >> input;


    
    

    while (input == "run" || input ==  "bench" || input == "verify") {


        if (input =="run") {
            //if this is the case
            //we will basically just start producing frames and pusing them onto the ring buffer
            jthread producer(produce);
        }

        if (input == "bench") {

        }

        if (input == "verify") {

        }
    }

    cout << "Please input run, bench, or verify" << endl;


}
