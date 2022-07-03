#include <iostream>
#include <vector>
#include <list>
#include <algorithm>
#include <stack>
#include <fstream>

using namespace std;
int num_loops = 0;

class state
{
public:
    vector<vector<bool>> workers;
    int num_workers;
    int days;
    vector<vector<bool>> visited;
    vector<string> level;
    vector<vector<int>> conflicts;
    int one_day_workers;
    state(int num_workers, int days, vector<int> level, vector<vector<int>> conflicts, int one_day_workers)
    {
        this->num_workers = num_workers;
        this->days = days;
        this->conflicts = conflicts;
        this->one_day_workers = one_day_workers;
        workers.resize(num_workers);
        visited.resize(num_workers);
        this->level.resize(num_workers);

        for (int i = 0; i < num_workers; i++)
        {
            workers[i].resize(days);
            visited[i].resize(days);
        }
        for (int i = 0; i < num_workers; i++)
        {
            for (int j = 0; j < days; j++)
            {
                workers[i][j] = false;
                visited[i][j] = false;
            }
            if (level[i] == 0)
            {
                this->level[i] = "junior";
            }
            else if (level[i] == 1)
            {
                this->level[i] = "senior";
            }
        }
    }
    state(state &s)
    {
        this->num_workers = s.num_workers;
        this->days = s.days;
        this->workers = s.workers;
        this->visited = s.visited;
        this->level = s.level;
        this->conflicts = s.conflicts;
        this->one_day_workers = s.one_day_workers;
    }
    bool is_constraint()
    {
        bool result = true;

        for (int i = 0; i < num_workers; i++)
        {
            result = result && rest_2days(workers[i], visited[i]);
            if (result == false)
            {
                /*
                cout << i << " cant rest 2 days"
                     << endl;*/
                return false;
            }
        }

        for (int i = 0; i < num_workers; i++)
        {
            result = result && cant_rest_3days(workers[i], visited[i]);
            if (result == false)
            {
                // cout << i << " three consecutive days off" << endl;
                return false;
            }
        }

        for (int j = 1; j <= days; j++)
        {
            result = result && less_One_senior_work(workers, visited, j);
            if (result == false)
            {
                // cout << j << " day no one senior work" << endl;
                return false;
            }
        }

        for (int j = 1; j <= days; j++)
        {
            result = result && less_workers(workers, visited, j);
            if (result == false)
            {
                // cout << j << " day not three workers" << endl;
                return false;
            }
        }
        result = result && cant_work_one_day();
        return result;
    }

    bool is_over()
    {
        //全部安排了工作
        for (int i = 0; i < num_workers; i++)
        {
            for (int j = 0; j < days; j++)
            {
                if (!visited[i][j])
                {
                    return false;
                }
            }
        }
        return true;
    }
    //每个工人必须休息两天
    bool rest_2days(vector<bool> &worker, vector<bool> &visited)
    {
        int nums = 0;
        int num_rest_days = 0;
        for (int i = 0; i < visited.size(); i++)
        {
            if (visited[i] == true)
            {
                nums++; //已分配的天数
                if (worker[i] == false)
                {
                    num_rest_days++;
                }
            }
        }
        //已分配6天
        if (nums == visited.size() - 1)
        {
            if (num_rest_days == 0)
            {
                return false;
            }
        }
        //已分配七天
        else if (nums == visited.size())
        {
            if (num_rest_days < 2)
                return false;
        }
        return true;
    }
    //每个工人不能连续休息三天
    bool cant_rest_3days(vector<bool> &worker, vector<bool> &visited)
    {
        int nums = 0;
        for (int i = 0; i < visited.size(); i++)
        {
            if (visited[i] == true)
            {
                nums++; //已分配的天数
            }
        }
        //不可连休三天
        if (nums >= 3 && nums < visited.size() && worker[nums - 1] == false && worker[nums - 2] == false && worker[nums - 3] == false)
        {
            return false;
        }
        return true;
    }
    //每天至少有一名senior的工人值班
    // day代表第几天
    bool less_One_senior_work(vector<vector<bool>> &worker, vector<vector<bool>> &visited, int day)
    {
        int nums_senior_works = 0;         //代表该天有多少个senior的工人值班
        int dont_visited_senior_works = 0; //代表未安排工作的senior工人数量
        for (int i = 0; i < num_workers; i++)
        {
            if (visited[i][day - 1] == true && level[i] == "senior" && worker[i][day - 1] == true)
            {
                nums_senior_works++;
            }
            if (visited[i][day - 1] == false && level[i] == "senior")
            {
                dont_visited_senior_works++;
            }
        }

        if (nums_senior_works == 0 && dont_visited_senior_works == 0)
        {
            return false;
        }

        return true;
    }
    //每天至少多少人值班
    bool less_workers(vector<vector<bool>> &worker, vector<vector<bool>> &visited, int day)
    {
        bool result = true;
        int nums_works = 0;           //代表该天有多少个工人值班
        int dont_visited_workers = 0; //代表未访问的工人
        for (int i = 0; i < num_workers; i++)
        {
            if (visited[i][day - 1] == true && workers[i][day - 1] == true)
            {
                nums_works++;
            }
            if (visited[i][day - 1] == false)
            {
                dont_visited_workers++;
            }
        }
        if (nums_works + dont_visited_workers < one_day_workers)
        {
            return false;
        }
        return true;
    }
    bool cant_work_one_day()
    {
        bool result = true;
        for (int day = 0; day < days; day++)
        {
            for (int i = 0; i < conflicts.size(); i++)
            {
                int worker1 = conflicts[i][0];
                int worker2 = conflicts[i][1];
                if (visited[worker1 - 1][day] == true && visited[worker2 - 1][day] == true)
                {
                    if (workers[worker1 - 1][day] == true && workers[worker2 - 1][day] == true)
                    {
                        result = false;
                        // cout<<"day:"<<day<<" conflict: "<<worker1<<" and "<<worker2<<endl;
                        return result;
                    }
                }
            }
        }
        return result;
    }
};

int main()
{
    // int num_workers = 7;
    // int days = 7;
    vector<vector<int>> cant_work_one_day;
    
    // 0代表junior,1代表senior
    //车间1
    ofstream outfile("../output/output1.txt");
    state *start_state = new state(7, 7, {1, 1, 0, 0, 0, 0, 0}, {{1, 4}, {2, 3}, {3, 6}}, 4);
    //车间2
    //ofstream outfile("../output/output2.txt");
    //state *start_state = new state(10, 7, {1, 1, 0, 0, 0, 0, 0,1,0,1},{{1,5},{2,6},{8,10}},5);
    stack<state *> s;
    s.push(start_state);
    while (!s.empty())
    {
        num_loops++;
        state *current = s.top();
        s.pop();
        if (current->is_over())
        {
            cout << "result:\n";
            // cout<<current->cant_work_one_day()<<endl;
            for(int j=0;j<current->days; j++)
            {
                for (int i = 0; i < current->num_workers; i++)
                {
                    if(current->workers[i][j])
                    {
                        cout<< i+1 <<" ";
                        outfile<<to_string(i+1)<<" ";
                    }
                    
                }
                cout<<endl;
                outfile<<endl;
            }
            break;
        }
        state *new_current1 = new state(*current);

        state *new_current2 = new state(*current);
        bool is_break = false;
        int num = 0;
        for (int i = 0; i < current->num_workers; i++)
        {
            for (int j = 0; j < current->days; j++)
            {
                if (current->visited[i][j] == false)
                {
                    new_current1->workers[i][j] = true;
                    new_current1->visited[i][j] = true;
                    new_current2->workers[i][j] = false;
                    new_current2->visited[i][j] = true;
                    num++;
                    is_break = true;
                }
                if (is_break)
                    break;
            }
            if (is_break)
                break;
        }

        bool is_ok1 = new_current1->is_constraint();
        bool is_ok2 = new_current2->is_constraint();
        if (is_ok1)
        {
            s.push(new_current1);
        }

        if (is_ok2)
        {
            s.push(new_current2);
        }
    }
    outfile.close();
    return 0;
}