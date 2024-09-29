// Boost.Geometry
// Unit Test

// Copyright (c) 2017 Oracle and/or its affiliates.

// Contributed and/or modified by Vissarion Fysikopoulos, on behalf of Oracle

// Use, modification and distribution is subject to the Boost Software License,
// Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_GEOMETRY_TEST_VERTEX_LONGITUDE_CASES_HPP
#define BOOST_GEOMETRY_TEST_VERTEX_LONGITUDE_CASES_HPP

struct coordinates
{
    double lon;
    double lat;
};

struct expected_result
{
    double lon;
    double lat;
};

struct expected_results
{
    coordinates p1;
    coordinates p2;
    double andoyer;
    double thomas;
    double vincenty;
    double spherical;
};

expected_results expected[] =
{ 
    { //ascenting segments (wrt pole)
      { 1, 1 },{ 100, 2 },
      66.25553538,
      66.25594187,
      66.25594273,
      66.39744208
    },{
        { 1, 1 },{ 90, 2 },
        64.09051435,
        64.09082287,
        64.09082352,
        64.24414382
    },{
        { 0, 1 },{ 50, 1 },
        24.99994265,
        24.99999906,
        25,
        25
    },{
        { 0, 1 },{ 50, 1.1 },
        30.79039009,
        30.79049758,
        30.79049828,
        30.83209056
    },{
        { 0, 1 },{ 50, 1.2 },
        35.95625867,
        35.95640445,
        35.95640493,
        36.0343576
    },{
        { 0, 1 },{ 50, 1.3 },
        40.52204781,
        40.52222064,
        40.52222094,
        40.63120292
    },{
        { 0, 1 },{ 50, 1.4 },
        44.53768967,
        44.53788045,
        44.53788061,
        44.6729585
    },{
        { 0, 1 },{ 50, 1.5 },
        48.06382018,
        48.0640219,
        48.06402195,
        48.22088385
    },{
        { 0, 1 },{ 50, 1.6 },
        50,
        50,
        50,
        50
    },{ //descending segment (wrt pole)
        { 50, 1 },{ 0, 1.1 },
        19.20950181,
        19.20950054,
        19.20950172,
        19.16790944
    },{
        { 50, 1 },{ 0, 1.2 },
        14.04365003,
        14.04359367,
        14.04359507,
        13.9656424
    },{
        { 50, 1 },{ 0, 1.3 },
        9.477883847,
        9.477777483,
        9.477779056,
        9.368797077
    },{
        { 50, 1 },{ 0, 1.4 },
        5.462267984,
        5.462117673,
        5.462119386,
        5.327041497
    },{
        { 50, 1 },{ 0, 1.5 },
        1.936164363,
        1.935976226,
        1.93597805,
        1.779116148
    },{
        { 3, 5 },{ 150, 0},
        60.29182988,
        60.29785309,
        60.29785255,
        60
    },{
        { 3, 5 },{ 150, 0.5},
        63.11344576,
        63.11900045,
        63.11899891,
        62.87000766
    },{
        { 3, 5 },{ 150, 1},
        65.51880171,
        65.52391866,
        65.52391623,
        65.31813729
    },{
        { 3, 5 },{ 150, 5},
        76.49727275,
        76.50000657,
        76.5,
        76.5
    },{ //segments parallel to equator
        { 0, 1 },{ 4, 1 },
        1.999999973,
        1.999999925,
        2,
        2
    },{
        { 0, 1 },{ 10, 1 },
        4.999999569,
        4.999999812,
        5,
        5
    },{
        { 0, 1 },{ 60, 1 },
        29.9998978,
        29.99999887,
        30,
        30
    },{
        { 0, 1 },{ 90, 1 },
        44.99960266,
        45.22272756,//thomas low accuracy
        45,
        45
    },{
        { 0, 1 },{ 120, 1 },
        59.99878311,
        59.99999778,
        60,
        60
    },{
        { 0, 1 },{ 180, 1 },
        90,
        90,
        90,
        90
    },{
        { 0, 1 },{ 270, 1 },
        -44.99960266,
        -45.08931472,//thomas low accuracy
        -45,
        -45
    },{
        { 0, 1 },{ 290, 1 },
        -34.9998314,
        -34.99999868,
        -35,
        -35
    },{
        { 0, 1 },{ 150, 1 },
        74.99598515,
        74.99999794,
        75,
        75
    },{
        { 0, 1 },{ 180, 1 },
        90,
        90,
        90,
        90
    },{ //in equator vertex is any point on segment
        { 1, 0 },{ 10, 0 },
        1,
        1,
        1,
        1
    },{// one point on equator (descending)
       { 150, 0},{ 3, 1 },
       60.29513726,
       60.30158943,
       60.3015947,
       60
    },{// one point on equator (ascending)
       { 3, 0 },{ 150, 1},
       92.69840523,
       92.6984053,
       92.6984053,
       93
    },{ //meridian
        { 1, 1 },{ 1, 2 },
        1,
        1,
        1,
        1
    },{ //nearly meridian
        { 1, 1 },{ 1.001, 2 },
        1.001,
        1.001,
        1.001,
        1.001
    },{ //vertex is a segment's endpoint
        { 1, 1 },{ 10, 2 },
        10,
        10,
        10,
        10
    },{
        { 10, 1 },{ 1, 2 },
        1,
        1,
        1,
        1
    },{ //South hemisphere, ascending
        { 0, -1 },{ 50, -1.4 },
        44.53768958,
        44.53788035,
        44.53788052,
        44.6729585
    },{ //South hemisphere, descending
        { 0, -1.5 },{ 50, -1 },
        1.936164356,
        1.935976219,
        1.935978042,
        1.779116148
    },{ //South hemisphere, same latitude
        { 0, -1 },{ 50, -1 },
        24.99994261,
        24.99999901,
        24.99999995,
        25
    },{//Both hemispheres, vertex on the northern
       //A desc vertex north
       { 3, 5 },{ 150, -3},
       27.357069,
       27.36422922,
       27.36423549,
       26.74999989
    },{//B asc vertex north
       { 3, -3 },{ 150, 5},
       125.6403436,
       125.6357677,
       125.6357659,
       126.2500001
    },{//C desc vertex south
       { 3, -5 },{ 150, 3},
       27.3570679,
       27.36422812,
       27.36423439,
       26.74999989
    },{//D asc vertex south
       { 3, 3 },{ 150, -5},
       125.6403423,
       125.6357664,
       125.6357645,
       126.2500001
    },{//E asc vertex south
       { 3, 3 },{ 184, -5},
       -88.00743796,
       -88.0660268,
       -88.0558747,
       -88.49315894
    },{
        { 3, 5 },{ 150, -3.5},
        17.96722293,
        17.97322407,
        17.97323051,
        17.3742464
    },{
        { 3, 5 },{ 150, -1},
        52.9706038,
        52.97759463,
        52.9775964,
        52.56504545
    },{ //Both hemispheres, vertex on the southern
        { 3, 3},{ 5, -5},
        5,
        5,
        5,
        5
    },{
        { 3, -5 },{ 150, 1}, //symmetric to { 3, 5 },{ 150, -1}
        52.97060093,
        52.97759176,
        52.97759353,
        52.56504545
    },{// fix p1 lon, lat and p2 lon and vary p2 lat
       { 3, 5 },{ 150, 1},
       65.51880171,
       65.52391866,
       65.52391623,
       65.31813729
    },{
        { 3, 5 },{ 150, 0},
        60.29182988,
        60.29785309,
        60.29785255,
        60
    },{
        { 3, 5 },{ 150, -0.1},
        59.66911673,
        59.67523649,
        59.67523616,
        59.36690727
    },{
        { 3, 5 },{ 150, -1},
        52.9706038,
        52.97759463,
        52.9775964,
        52.56504545
    },{
        { 3, 5 },{ 150, -4.15},
        4.481947557,
        4.485467841,
        4.485473295,
        3.981178967
    },{
        { 3, 5 },{ 150, -4.2},
        3,
        3,
        3,
        3
    },{//symmetry of geodesics:
           // (i) case A same as C and B same as D
           // (ii) longitude diff between vertex and p2 in A, C equals
           //      longitude diff between vertex and p1 in B, D by symmetry
           // case (A)
           { 0, 5 },{ 30, 5.5},
           25.06431998,
           25.0644277,
           25.06442787,
           25.13253724
        },{// case (B)
           { 0, 5.5 },{ 30, 5},
           4.935667094,
           4.935571216,
           4.93557213,
           4.867462762
        },{// case (C)
           { 0, -5 },{ 30, -5.5},
           25.06431885,
           25.06442657,
           25.06442674,
           25.13253724
        },{// case (D)
           { 0, -5.5 },{ 30, -5},
           4.935666841,
           4.935570963,
           4.935571877,
           4.867462762
        },{//crossing meridian
           { -10, 1 },{ 50, 1.1},
           24.68113946,
           24.68127641,
           24.68127733,
           24.71605263
        },{
           { 350, 1 },{ 50, 1.1},
           24.68113946,
           24.68127641,
           24.68127733,
           24.71605263
        },{//crossing antimeridian
           { 130, 1 },{ 190, 1.1},
           164.6811395,
           164.6812764,
           164.6812773,
           164.7160526
        },{
           { 130, 1 },{ -170, 1.1},
           164.6811395,
           164.6812764,
           164.6812773,
           164.7160526
        },{//crossing meridian both hemispheres
           { -10, -5 },{ 150, 1},
           55.61285835,
           55.62727853,
           55.62725182,
           55.19943725
        },{
            { 350, -5 },{ 150, 1},
            55.6243632,
            55.6272619,
            55.627257,
            55.1994373
        },{//crossing anti-meridian both hemispheres
           { 90, -5 },{ 210, 1},
           109.4997596,
           109.5011987,
           109.5012031,
           109.1354089
        },{
            { 90, -5 },{ -150, 1},
            109.4997596,
            109.5011987,
            109.5012031,
            109.1354089
        },{
            { -150, -5 },{ 90, 1},
            -169.4997596,
            -169.5011987,
            -169.5012031,
            -169.1354089
        },{
            { 90, 1 },{ 210, -5},
            -169.5008004,
            -169.5012037,
            -169.501204,
            -169.1354089
        },{
            { 0, 1 },{ 120, -5},
            100.4991996,
            100.4987963,
            100.498796,
            100.8645911
        }
};

size_t const expected_size = sizeof(expected) / sizeof(expected_results);

#endif // BOOST_GEOMETRY_TEST_VERTEX_LONGITUDE_CASES_HPP
