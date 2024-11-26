#include <bits/stdc++.h>
using namespace std;

int main(){

  int num = 0b0001010100;
  int mutated_num = num ^ (1 << 6); //XOR

  0001010100
  0010000000
  0011010100

  cout << num << "\n";
  cout << mutated_num << "\n";
  return 0;
}
