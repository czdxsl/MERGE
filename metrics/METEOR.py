from py_meteor.meteor import Meteor

meteor = Meteor()


def meteor_m(reference, candidate):
    score = meteor.meteor_score(candidate, reference)

    return score
