
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Helpers.h"
#include "MPC.h"
#include "json.hpp"
#include "Constants.h"

// for convenience
using json = nlohmann::json;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(const std::string& s)
{
    auto found_null = s.find("null");
    auto b1 = s.find_first_of("[");
    auto b2 = s.rfind("}]");
    if (found_null != std::string::npos)
    {
        return "";
    } else if (b1 != std::string::npos && b2 != std::string::npos)
    {
        return s.substr(b1, b2 - b1 + 2);
    }
    return "";
}

void convertWaypointsToCarCoordinate(
    const std::vector<double>& pointsX,
    const std::vector<double>& pointsY,
    double carX,
    double carY,
    double carPsi,
    Eigen::VectorXd& pointsInCarCoordX,
    Eigen::VectorXd& pointsInCarCoordY)
{
    assert(pointsX.size() == pointsY.size());

    const size_t numPoints = pointsX.size();
    for (size_t index = 0; index < numPoints; index++)
    {
        double localX = 0;
        double localY = 0;
        Helpers::convertToCarCoordinate(carX, carY, carPsi, pointsX[index], pointsY[index], localX, localY);

        pointsInCarCoordX[index] = localX;
        pointsInCarCoordY[index] = localY;
    }
}

std::vector<double> getActuationsFromMPC(const Eigen::VectorXd& state, const Eigen::VectorXd& coeffs, double& steer, double& throttle)
{
    double delta = 0;
    double a = 0;
    std::vector<double> predictedTrajectory = MPC::Solve(state, coeffs, delta, a);

    // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
    // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
    steer = - Helpers::clamp(delta / Helpers::deg2rad(25.0), -1.0, 1.0);
    throttle = Helpers::clamp(a, -1.0, 1.0);

    return predictedTrajectory;
}

void displayFitedTrajectory(json& msgJson, double v, const Eigen::VectorXd& coeffs)
{
    std::vector<double> next_x_vals;
    std::vector<double> next_y_vals;

    for (size_t index = 0; index < Constants::N; index++)
    {
        const double x = Constants::DT * v * static_cast<double>(index);
        const double y = Helpers::polyeval(coeffs, x);

        next_x_vals.emplace_back(x);
        next_y_vals.emplace_back(y);
    }

    //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
    // the points in the simulator are connected by a Yellow line
    msgJson["next_x"] = next_x_vals;
    msgJson["next_y"] = next_y_vals;
}

void displayPredictedTrajectory(json& msgJson, const std::vector<double>& predictedTrajectory)
{
    //Display the MPC predicted trajectory
    std::vector<double> mpc_x_vals;
    std::vector<double> mpc_y_vals;

    for (size_t index = 0; index < Constants::N; index++)
    {
        mpc_x_vals.emplace_back(predictedTrajectory[2 * index]);
        mpc_y_vals.emplace_back(predictedTrajectory[2 * index + 1]);
    }

    //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
    // the points in the simulator are connected by a Green line
    msgJson["mpc_x"] = mpc_x_vals;
    msgJson["mpc_y"] = mpc_y_vals;
}

Eigen::VectorXd predictState(
    const Eigen::VectorXd& state,
    double steering,
    double throttle,
    double time)
{
    const double x = state[0];
    const double y = state[1];
    const double psi = state[2];
    const double v = state[3];
    const double cte = state[4];
    const double epsi = state[5];
    const double a = throttle;
    const double delta = -steering * Helpers::deg2rad(25.0);

    const double newX = x + v * cos(psi) * time;
    const double newY = y + v * sin(psi) * time;
    const double newPsi = psi + (v/Constants::Lf) * delta * time;
    const double newV = v + a * time;
    const double newCte = cte + v * sin(epsi) * time;
    const double newEpsi = epsi + (v / Constants::Lf) * delta * time;

    Eigen::VectorXd predictedState(Constants::STATE_DIM);
    predictedState << newX, newY, newPsi, newV, newCte, newEpsi;
    return predictedState;
}

int main() {
    uWS::Hub h;

    h.onMessage([](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode)
    {
        // "42" at the start of the message means there's a websocket message event.
        // The 4 signifies a websocket message
        // The 2 signifies a websocket event
        std::string sdata = std::string(data).substr(0, length);
        std::cout << sdata << std::endl;
        if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2')
        {
            std::string s = hasData(sdata);
            if (s.empty())
            {
                // Manual driving
                std::string msg = "42[\"manual\",{}]";
                ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
                return;
            }

            auto j = json::parse(s);
            std::string event = j[0].get<std::string>();
            if (event != "telemetry")
            {
                return;
            }

            // j[1] is the data JSON object
            double px = j[1]["x"];
            double py = j[1]["y"];
            double psi = j[1]["psi"];
            double v = j[1]["speed"];
            double delta = j[1]["steering_angle"];
            double a = j[1]["throttle"];
            const std::vector<double>& globalPtsX = j[1]["ptsx"];
            const std::vector<double>& globalPtsY = j[1]["ptsy"];

            // convert all way points to the car coordinate 
            const size_t numPoints = globalPtsX.size();
            Eigen::VectorXd ptsx(numPoints);
            Eigen::VectorXd ptsy(numPoints);
            convertWaypointsToCarCoordinate(globalPtsX, globalPtsY, px, py, psi, ptsx, ptsy);

            Eigen::VectorXd coeffs = Helpers::polyfit(ptsx, ptsy, 2);
            std::cout << "Coeffs : " << coeffs << std::endl;

            // after converting to car coordinate, px, py and psi all become 0
            px = 0;
            py = 0;
            psi = 0;

            double cte = Helpers::polyeval(coeffs, px) - py;
            double epsi = psi - Helpers::getPsiTarget(coeffs, px);

            Eigen::VectorXd state(6);
            state << px, py, psi, v, cte, epsi;

            // due to the delay introduced, first predicte the state after the latency and use the new state as the input
            Eigen::VectorXd inputState = predictState(state, delta, a, 0.1);

            double steer_value = 0;
            double throttle_value = 0;
            std::vector<double> predictedTrajectory = getActuationsFromMPC(inputState, coeffs, steer_value, throttle_value);
            std::cout << "Steering is: " << steer_value << ", Throttle is:" << throttle_value << std::endl;

            json msgJson;
            msgJson["steering_angle"] = steer_value;
            msgJson["throttle"] = throttle_value;

            displayPredictedTrajectory(msgJson, predictedTrajectory);
            displayFitedTrajectory(msgJson, v, coeffs);

            auto msg = "42[\"steer\"," + msgJson.dump() + "]";
            //std::cout << msg << std::endl;

            // Latency
            // The purpose is to mimic real driving conditions where
            // the car does actuate the commands instantly.
            //
            // Feel free to play around with this value but should be to drive
            // around the track with 100ms latency.
            //
            // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
            // SUBMITTING.
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
    });

    // We don't need this since we're not using HTTP but if it's removed the
    // program, doesn't compile :-(
    h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t)
    {
        const std::string s = "<h1>Hello world!</h1>";
        if (req.getUrl().valueLength == 1)
        {
            res->end(s.data(), s.length());
        }
        else
        {
            // i guess this should be done more gracefully?
            res->end(nullptr, 0);
        }
    });

    h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req)
    {
        std::cout << "Connected!!!" << std::endl;
    });

    h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length)
    {
        ws.close();
        std::cout << "Disconnected" << std::endl;
    });

    int port = 4567;
    if (h.listen(port))
    {
        std::cout << "Listening to port " << port << std::endl;
    }
    else
    {
        std::cerr << "Failed to listen to port" << std::endl;
        return -1;
    }
    h.run();
}
