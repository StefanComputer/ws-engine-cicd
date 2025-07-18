#pragma once

#include "LibAiExportOs.h"
#include "Scriptable.h"
#include "NamedGroup.h"
#include <string>
#include <vector>

namespace WS
{
  /// @brief JsonConfig is used to store the different parameters sent from the server.
  class LIB_AI_EXPORT JsonConfigEQ : public WS::Util::Scriptable
  {
  public:
    JsonConfigEQ();
    virtual ~JsonConfigEQ();

    struct FilterData
    {
      // The EQ filter data
      std::string mType;
      u32 mFrequency;
      float mQ;
      float mGain;

      FilterData()
          : mType{"Peak"}, mFrequency{100}, mQ{1.F}, mGain{1.F}
      {
      }
    };

    /// Scriptable overrides
    bool readFromStream(WS::Util::JsonReader &) override;
    bool writeToStream(WS::Util::JsonWriter &) override;

  public:
    /// Inputs to StreamerManager process
    float mModelParamValue;
    bool mApplyEq;
    bool mApplyPreEq;
    int mNumEqFilters;

    std::vector<FilterData> mEQFilters;
    WS::Util::NamedGroup mEQFilterGrp;

  private:
    void resizeEQFilters(u32 filterCnt)
    {
      mEQFilters.clear();
      mEQFilters.resize(filterCnt);

      mEQFilterGrp.clear();
    }
  };
} // WS