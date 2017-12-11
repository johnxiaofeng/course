#include "math_utils.h"
#include <cmath>

bool MathUtils::DoubleEqual(double left, double right, double delta)
{
    return fabs(left - right) < delta;
}
