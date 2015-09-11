#include <iostream>
#include "PCA.h"

int main()
{
    PCA pcasm("/Users/liuyang/Desktop/Normfx", 2, 400);
    
    std::cout << pcasm.Whitening();
}