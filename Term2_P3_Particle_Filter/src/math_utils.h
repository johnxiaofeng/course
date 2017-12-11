#ifndef MATH_UTILS
#define MATH_UTILS

class MathUtils
{
public:
    /**
    * A helper method to compare if two float values are equal
    */
    static bool DoubleEqual(double left, double right, double delta = 0.00001);
};

#endif //MATH_UTILS