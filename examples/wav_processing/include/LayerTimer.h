#ifndef _WS_LAYERTIMER_H
#define _WS_LAYERTIMER_H

#include <chrono>
#include <iostream>

//#define BENCHMARKS //Comments to disable benchmarks

namespace WS
{
class LayerTimer
{
public:
    LayerTimer() = default;
    virtual ~LayerTimer() = default;
    
    inline void startTimer()
    {
#ifdef BENCHMARKS
        t1 = std::chrono::high_resolution_clock::now();
#endif
    }

#ifdef BENCHMARKS
    inline void endTimer(std::string const& layer)
    {
        t2 = std::chrono::high_resolution_clock::now();
        float time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
        mTotalTime += time_taken;
        printMessage(layer, time_taken);
    }
#else
    inline void endTimer(std::string const&)
    {
    }
#endif

#ifdef BENCHMARKS
    inline void printMessage(std::string const& layer, float time)
    {
        std::cout << "Layer "<<layer << " took " << time << " ms " << std::endl;
    }
#else
    inline void printMessage(std::string const&, float)
    {
    }
#endif

    inline void printTotalTime(){
#ifdef BENCHMARKS
        std::cout << "Lib-Ai  took " << mTotalTime << " ms " << std::endl;
#endif
    }
private:
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
#ifdef BENCHMARKS
    float mTotalTime = 0.0f;
#endif
};
}
#endif
