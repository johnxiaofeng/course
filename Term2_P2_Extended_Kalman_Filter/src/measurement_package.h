#ifndef MEASUREMENT_PACKAGE_H_
#define MEASUREMENT_PACKAGE_H_

#include "types.h"

struct MeasurementPackage 
{
    enum SensorType
    {
        LASER,
        RADAR
    };

    long long timestamp_;
    SensorType sensor_type_;
    Vector raw_measurements_;
};

#endif //MEASUREMENT_PACKAGE_H_
