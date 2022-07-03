#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <queue>
#include <cstdio>
#include <ctime>
#include <climits>
#include <stack>
#include <algorithm>
using namespace std;

class state
{
protected:
    vector<vector<int>> location;
    vector<vector<int>> target;
    char last_move;
    int h;
    int g;
    bool valid;
    const state *last_state;
    int d;

public:
    state(const vector<vector<int>> &location, const vector<vector<int>> &target, const int &g)
    {
        this->location = location;
        this->target = target;
        this->last_move = '#';
        // this->h = h;
        this->g = g;
        this->valid = true;
        last_state = nullptr;
        this->set_h();
        this->d = this->get_g() + this->get_h();
    }
    state(const state &s)
    {
        this->location = s.location;
        this->target = s.target;
        this->last_move = s.last_move;
        this->h = s.h;
        this->g = s.g;
        this->valid = s.valid;
        this->last_state = s.last_state;
        this->d = s.d;
    }
    const state *get_last_state() const
    {
        return this->last_state;
    }
    vector<vector<int>> get_loaction() const
    {
        return this->location;
    }
    char get_last_move() const
    {
        return this->last_move;
    }
    int get_h() const
    {
        return this->h;
    }
    int get_g() const
    {
        return this->g;
    }
    bool get_valid() const
    {
        return this->valid;
    }
    bool is_over() const
    {
        return this->location == this->target;
    }
    int get_d() const
    {
        return this->d;
    }

    int compute_h() const
    {
        int num_misplaced = 0;
        for (int i = 0; i < location.size(); i++)
        {

            for (int j = 0; j < location[i].size(); j++)
            {

                if (location[i][j] != 0 && (location[i][j] != target[i][j]))
                {
                    num_misplaced++;
                }
            }
        }

        return num_misplaced;
    }
    void set_h()
    {
        h = compute_h();
        d = h + g;
    }
    void findAirship(int &i, int &j) const
    {
        for (i = 0; i < location.size(); i++)
        {
            for (j = 0; j < location[i].size(); j++)
            {
                if (location[i][j] == 0)
                {
                    // cout<<"i="<<i<<"   j="<<j<<endl;
                    return;
                }
            }
        }
    }
    void findBlackHole(int &i, int &j) const
    {
        for (i = 0; i < location.size(); i++)
        {
            for (j = 0; j < location[i].size(); j++)
            {
                if (location[i][j] < 0)
                {
                    // cout<<"i="<<i<<"   j="<<j<<endl;
                    return;
                }
            }
        }
    }

    state *move(char direction) const
    {
        state *temp = new state(*this);

        int i = 0, j = 0;
        findAirship(i, j);
        int k = 0, l = 0;
        findBlackHole(k, l);
        temp->last_move = direction;
        temp->last_state = this;
        // temp->d = temp->get_g() + temp->get_h();
        switch (direction)
        {
        case 'R':
        {
            //判断边界
            if ((j == 4 && i != 2) || (i == k && j + 1 == l))
            {
                temp->valid = false;
                break;
            }
            //特殊处理隧道
            else if (j == 4 && i == 2)
            {
                int temp_val = temp->location[2][0];
                temp->location[2][0] = temp->location[i][j];
                temp->location[i][j] = temp_val;
                break;
            }
            else
            {
                int temp_val = temp->location[i][j];
                temp->location[i][j] = temp->location[i][j + 1];
                temp->location[i][j + 1] = temp_val;
                break;
            }
            break;
        }
        case 'L':
        {
            if ((j == 0 && i != 2) || (i == k && j - 1 == l))
            {
                temp->valid = false;
                break;
            }
            else if (j == 0 && i == 2)
            {
                int temp_val = temp->location[2][4];
                temp->location[2][4] = temp->location[i][j];
                temp->location[i][j] = temp_val;
                break;
            }
            else
            {
                int temp_val = temp->location[i][j];
                temp->location[i][j] = temp->location[i][j - 1];
                temp->location[i][j - 1] = temp_val;
                break;
            }
            break;
        }
        case 'U':
        {

            if ((i == 0 && j != 2) || (i - 1 == k && j == l))
            {
                temp->valid = false;
            }
            else if (i == 0 && j == 2)
            {
                int temp_val = temp->location[4][2];
                temp->location[4][2] = temp->location[i][j];
                temp->location[i][j] = temp_val;
            }
            else
            {
                int temp_val = temp->location[i][j];
                temp->location[i][j] = temp->location[i - 1][j];
                temp->location[i - 1][j] = temp_val;
            }

            break;
        }
        case 'D':
        {
            if ((i == 4 && j != 2) || (i + 1 == k && j == l))
            {
                temp->valid = false;
                break;
            }
            else if (i == 4 && j == 2)
            {
                int temp_val = temp->location[0][2];
                temp->location[0][2] = temp->location[i][j];
                temp->location[i][j] = temp_val;
                break;
            }
            else
            {
                int temp_val = temp->location[i][j];
                temp->location[i][j] = temp->location[i + 1][j];
                temp->location[i + 1][j] = temp_val;
                break;
            }
            break;
        }
        }

        temp->g = temp->g + 1;

        // int misplaced_stars = temp->get_misplaced_stars();
        temp->set_h();

        return temp;
    }
};

class sub_state
{
protected:
    vector<vector<int>> location;
    vector<vector<int>> target;
    char last_move;
    int h;
    int g;
    bool valid;
    const sub_state *last_state;
    int d;

public:
    sub_state(const vector<vector<int>> &location, const vector<vector<int>> &target, const int &g)
    {
        this->location = location;
        this->target = target;
        this->last_move = '#';
        this->g = g;
        this->valid = true;
        last_state = nullptr;
        this->set_h();
        this->d = this->get_g() + this->get_h();
    }
    sub_state(const sub_state &s)
    {
        this->location = s.location;
        this->target = s.target;
        this->last_move = s.last_move;
        this->h = s.h;
        this->g = s.g;
        this->valid = s.valid;
        this->last_state = s.last_state;
        this->d = s.d;
    }
    const sub_state *get_last_state() const
    {
        return this->last_state;
    }
    vector<vector<int>> get_loaction() const
    {
        return this->location;
    }
    char get_last_move() const
    {
        return this->last_move;
    }
    int get_h() const
    {
        return this->h;
    }
    int get_g() const
    {
        return this->g;
    }
    bool get_valid() const
    {
        return this->valid;
    }
    bool is_over() const
    {
        return this->location == this->target;
    }
    int get_d() const
    {
        return this->d;
    }

    void set_h()
    {
        h = compute_h();
        d = h + g;
    }
    void findAirship(int &i, int &j) const
    {
        for (i = 0; i < location.size(); i++)
        {
            for (j = 0; j < location[i].size(); j++)
            {
                if (location[i][j] == 0)
                {
                    // cout<<"i="<<i<<"   j="<<j<<endl;
                    return;
                }
            }
        }
    }
    void findBlackHole(int &i, int &j) const
    {
        for (i = 0; i < location.size(); i++)
        {
            for (j = 0; j < location[i].size(); j++)
            {
                if (location[i][j] < 0)
                {
                    // cout<<"i="<<i<<"   j="<<j<<endl;
                    return;
                }
            }
        }
    }

    sub_state *move(char direction) const
    {
        sub_state *temp = new sub_state(*this);

        int i = 0, j = 0;
        findAirship(i, j);
        int k = 0, l = 0;
        findBlackHole(k, l);
        temp->last_move = direction;
        temp->last_state = this;
        // temp->d = temp->get_g() + temp->get_h();
        switch (direction)
        {
        case 'R':
        {
            if ((j == 4 && i != 2) || (i == k && j + 1 == l))
            {
                temp->valid = false;
                break;
            }
            else if (j == 4 && i == 2)
            {
                int temp_val = temp->location[2][0];
                temp->location[2][0] = temp->location[i][j];
                temp->location[i][j] = temp_val;
                break;
            }
            else
            {
                int temp_val = temp->location[i][j];
                temp->location[i][j] = temp->location[i][j + 1];
                temp->location[i][j + 1] = temp_val;
                break;
            }
            break;
        }
        case 'L':
        {
            if ((j == 0 && i != 2) || (i == k && j - 1 == l))
            {
                temp->valid = false;
                break;
            }
            else if (j == 0 && i == 2)
            {
                int temp_val = temp->location[2][4];
                temp->location[2][4] = temp->location[i][j];
                temp->location[i][j] = temp_val;
                break;
            }
            else
            {
                int temp_val = temp->location[i][j];
                temp->location[i][j] = temp->location[i][j - 1];
                temp->location[i][j - 1] = temp_val;
                break;
            }
            break;
        }
        case 'U':
        {

            if ((i == 0 && j != 2) || (i - 1 == k && j == l))
            {
                temp->valid = false;
            }
            else if (i == 0 && j == 2)
            {
                int temp_val = temp->location[4][2];
                temp->location[4][2] = temp->location[i][j];
                temp->location[i][j] = temp_val;
            }
            else
            {
                int temp_val = temp->location[i][j];
                temp->location[i][j] = temp->location[i - 1][j];
                temp->location[i - 1][j] = temp_val;
            }

            break;
        }
        case 'D':
        {
            if ((i == 4 && j != 2) || (i + 1 == k && j == l))
            {
                temp->valid = false;
                break;
            }
            else if (i == 4 && j == 2)
            {
                int temp_val = temp->location[0][2];
                temp->location[0][2] = temp->location[i][j];
                temp->location[i][j] = temp_val;
                break;
            }
            else
            {
                int temp_val = temp->location[i][j];
                temp->location[i][j] = temp->location[i + 1][j];
                temp->location[i + 1][j] = temp_val;
                break;
            }
            break;
        }
        }

        temp->g = temp->g + 1;

        // int misplaced_stars = temp->get_misplaced_stars();
        temp->set_h();

        return temp;
    }

    int Manhattan_distance(int i, int j, int k, int l) const
    {
        return abs(i - k) + abs(j - l);
    }
    void find_otherStar(int value, int &k, int &l) const
    {
        for (int i = 0; i < target.size(); i++)
        {
            for (int j = 0; j < target[i].size(); j++)
            {
                if (target[i][j] == value)
                {
                    k = i;
                    l = j;
                    return;
                }
            }
        }
    }
    int compute_h() const
    {
        int result = 0;
        int k = 0, l = 0;
        int distance = 0;
        int distance_up = 0, distance_down = 0, distance_left = 0, distance_right = 0;
        for (int i = 0; i < location.size(); i++)
        {
            for (int j = 0; j < location[i].size(); j++)
            {
                if (location[i][j] > 0)
                {
                    find_otherStar(location[i][j], k, l);
                    distance = Manhattan_distance(i, j, k, l);
                    distance_up = Manhattan_distance(i, j, 0, 2) + 1 + Manhattan_distance(4, 2, k, l);
                    distance_down = Manhattan_distance(i, j, 4, 2) + 1 + Manhattan_distance(0, 2, k, l);
                    distance_left = Manhattan_distance(i, j, 2, 0) + 1 + Manhattan_distance(2, 4, k, l);
                    distance_right = Manhattan_distance(i, j, 2, 4) + 1 + Manhattan_distance(2, 0, k, l);
                    vector<int> temp = {distance, distance_up, distance_down, distance_left, distance_right};
                    result += *min_element(temp.begin(), temp.end());
                }
            }
        }
        return result;
    }
};