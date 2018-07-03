#pragma once

class Collision_freeGoal : public bio_ik::Goal {
    std::vector<std::string> mLinkNames = {{
        "rh_fftip", "rh_ffdistal", "rh_ffmiddle", "rh_ffproximal",
        "rh_mftip", "rh_mfdistal", "rh_mfmiddle", "rh_mfproximal",
        "rh_rftip", "rh_rfdistal", "rh_rfmiddle", "rh_rfproximal",
        "rh_lftip", "rh_lfdistal", "rh_lfmiddle", "rh_lfproximal",
        "rh_thtip", "rh_thdistal", "rh_thmiddle", "rh_thproximal",
    }};
    std::vector<std::string> mtipLinks = {{
        "rh_fftip",
        "rh_mftip",
        "rh_rftip",
        "rh_lftip",
        "rh_thtip",
    }};

    std::vector<std::array<size_t, 2>> mCollisionPairs{{
        {{0, 4}},
        {{4, 8}},
        {{8, 12}},
        {{12, 16}},
        {{16, 0}},

        {{0+1, 4+1}},
        {{4+1, 8+1}},
        {{8+1, 12+1}},
        {{12+1, 16+1}},
        {{16+1, 0+1}},

        {{0+2, 4+2}},
        {{4+2, 8+2}},
        {{8+2, 12+2}},
        {{12+2, 16+2}},
        {{16+2, 0+2}},

        {{0+3, 4+3}},
        {{4+3, 8+3}},
        {{8+3, 12+3}},
        {{12+3, 16+3}},
        {{16+3, 0+3}},

        {{0, 4+1}},
        {{4, 8+1}},
        {{8, 12+1}},
        {{12, 16+1}},
        {{16, 0+1}},

        {{0+1, 4}},
        {{4+1, 8}},
        {{8+1, 12}},
        {{12+1, 16}},
        {{16+1, 0}},

        {{0, 4+2}},
        {{4, 8+2}},
        {{8, 12+2}},
        {{12, 16+2}},
        {{16, 0+2}},

        {{0+2, 4}},
        {{4+2, 8}},
        {{8+2, 12}},
        {{12+2, 16}},
        {{16+2, 0}},

        {{0+3, 4+2}},
        {{4+3, 8+2}},
        {{8+3, 12+2}},
        {{12+3, 16+2}},
        {{16+3, 0+2}},

        {{0+2, 4+3}},
        {{4+2, 8+3}},
        {{8+2, 12+3}},
        {{12+2, 16+3}},
        {{16+2, 0+3}},
    }};
    double mCollisionRadius = 0.015;
    size_t mLinkCount = mLinkNames.size();

protected:
    double weight_;

public:
    Collision_freeGoal(double weight = 1.0):weight_(weight)
    {
    }
    virtual void describe(bio_ik::GoalContext &context) const override {
      Goal::describe(context);
      context.setWeight(weight_);
      for (auto &linkName : mLinkNames)
        context.addLink(linkName);
    }
    virtual double evaluate(const bio_ik::GoalContext &context) const override {
      double cost = 0.0;

      for (auto &p : mCollisionPairs) {
        double d = context.getLinkFrame(p[0]).getPosition().distance(
          context.getLinkFrame(p[1]).getPosition());
          // d is the bigger the better
          d = std::max(0.0, mCollisionRadius * 2 - d);
          cost += d * d * 8;
      }
      return cost;
    }
};
