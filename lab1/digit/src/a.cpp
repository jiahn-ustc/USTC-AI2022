
#include "a.hpp"

using namespace std;

struct cmp
{
    bool operator()(state *a, state *b)
    {
        return a->get_h() + a->get_g() > b->get_h() + b->get_g();
    }
};
struct sub_cmp
{
    bool operator()(sub_state *a, sub_state *b)
    {
        return a->get_h() + a->get_g() > b->get_h() + b->get_g();
    }
};

void A_h1(const vector<vector<int>> &start, const vector<vector<int>>
                                                &target)
{
    cout << "A_h1" << endl;

    //ofstream outfile;
    //outfile.open("../output/output_A_h1.txt",ios::app);
    priority_queue<state *, vector<state *>, cmp> open;
    string close;
    clock_t start_time, end_time;
    start_time = clock();
    state *start_state = new state(start, target, 0);
    start_state->set_h();

    open.push(start_state);
    while (!open.empty())
    {
        state *current = open.top();
        open.pop();
        vector<vector<int>> s = current->get_loaction();
        //判断是否结束
        if (current->is_over())
        {
            const state *temp = current;
            //cout << "d=" << temp->get_d() << endl;
            while (temp->get_last_state() != NULL)
            {
                close += temp->get_last_move();
                temp = temp->get_last_state();
            }
            for (int i = close.size() - 1; i >= 0; i--)
            {
                cout << close[i];
            }
            cout << endl;
            break;
        }
        //四个后继状态
        state *up_state = current->move('U');
        state *down_state = current->move('D');
        state *left_state = current->move('L');
        state *right_state = current->move('R');
        if (up_state->get_valid())
        {
            open.push(up_state);
        }
        if (down_state->get_valid())
        {
            open.push(down_state);
        }
        if (left_state->get_valid())
        {
            open.push(left_state);
        }
        if (right_state->get_valid())
        {
            open.push(right_state);
        }
    }
    end_time = clock();
    cout << "time = " << double(end_time - start_time) / CLOCKS_PER_SEC << "s" << endl;
    //reverse(close.begin(), close.end());
    //outfile <<close<<","<<to_string(double(end_time - start_time) / CLOCKS_PER_SEC)<<endl;
}
void A_h2(const vector<vector<int>> &start, const vector<vector<int>>
                                                &target)
{
    cout << "A_h2" << endl;
    //ofstream outfile;
    //outfile.open("../output/output_A_h2.txt",ios::app);
    priority_queue<sub_state *, vector<sub_state *>, sub_cmp> open;
    string close;
    clock_t start_time, end_time;
    start_time = clock();
    sub_state *start_state = new sub_state(start, target, 0);
    start_state->set_h();

    open.push(start_state);
    while (!open.empty())
    {
        sub_state *current = open.top();
        open.pop();
        vector<vector<int>> s = current->get_loaction();
        if (current->is_over())
        {
            const sub_state *temp = current;
            cout << "d=" << temp->get_d() << endl;
            while (temp->get_last_state() != NULL)
            {
                close += temp->get_last_move();
                temp = temp->get_last_state();
            }
            for (int i = close.size() - 1; i >= 0; i--)
            {
                cout << close[i];
            }
            cout << endl;
            break;
        }
        sub_state *up_state = current->move('U');
        sub_state *down_state = current->move('D');
        sub_state *left_state = current->move('L');
        sub_state *right_state = current->move('R');
        if (up_state->get_valid())
        {
            open.push(up_state);
        }
        if (down_state->get_valid())
        {
            open.push(down_state);
        }
        if (left_state->get_valid())
        {
            open.push(left_state);
        }
        if (right_state->get_valid())
        {
            open.push(right_state);
        }
    }
    end_time = clock();
    cout << "time = " << double(end_time - start_time) / CLOCKS_PER_SEC << "s" << endl;
    //reverse(close.begin(), close.end());
    //outfile <<close<<","<<to_string(double(end_time - start_time) / CLOCKS_PER_SEC)<<endl;
}
void IDA_h1(const vector<vector<int>> &start, const vector<vector<int>>
                                                  &target)
{
    cout << "IDA_h1" << endl;
    //ofstream outfile;
    //outfile.open("../output/output_IDA_h1.txt",ios::app);
    stack<state *> open;
    
    string close;
    state *start_state = new state(start, target, 0);
    int d_limit = start_state->get_d();
    bool is_end = false;
    clock_t start_time, end_time;
    start_time = clock();
    int is_start = 0;
    while (d_limit < INT_MAX)
    {
        cout << "d_limit: " << d_limit << endl;
        //每一次循环开始都要清空open
        while (!open.empty())
        {
            state *current = open.top();
            open.pop();
            delete current;
        }
        if (is_start == 0)
        {
            is_start = 1;
        }
        else
        {
            start_state = new state(start, target, 0);
        }
        int next_d_limit = INT_MAX;
        open.push(start_state);
        while (!open.empty())
        {
            state *current = open.top();
            open.pop();
            if (current->get_d() > d_limit)
            {

                next_d_limit = min(next_d_limit, current->get_d());
            }
            else
            {
                if (current->is_over())
                {
                    const state *temp = current;
                    while (temp->get_last_state() != NULL)
                    {
                        close += temp->get_last_move();
                        temp = temp->get_last_state();
                    }
                    for (int i = close.size() - 1; i >= 0; i--)
                    {
                        cout << close[i];
                    }
                    cout << endl;
                    end_time = clock();
                    cout << "time = " << double(end_time - start_time) / CLOCKS_PER_SEC << "s" << endl;
                    is_end = true;
                    break;
                }
                state *up_state = current->move('U');
                state *down_state = current->move('D');
                state *left_state = current->move('L');
                state *right_state = current->move('R');
                if (up_state->get_valid())
                {
                    open.push(up_state);
                }
                if (down_state->get_valid())
                {
                    open.push(down_state);
                }
                if (left_state->get_valid())
                {
                    open.push(left_state);
                }
                if (right_state->get_valid())
                {
                    open.push(right_state);
                }
            }
        }
        d_limit = next_d_limit;
        if (is_end)
            break;
    }
    //reverse(close.begin(), close.end());
    //outfile <<close<<","<<to_string(double(end_time - start_time) / CLOCKS_PER_SEC)<<endl;
}
void IDA_h2(const vector<vector<int>> &start, const vector<vector<int>>
                                                  &target)
{
    cout << "IDA_h2" << endl;
    //ofstream outfile;
    //outfile.open("../output/output_IDA_h2.txt",ios::app);
    stack<sub_state *> open;
    // priority_queue<sub_state *, vector<sub_state *>, cmp> open;
    string close;
    sub_state *start_state = new sub_state(start, target, 0);
    int d_limit = start_state->get_d();
    bool is_end = false;
    clock_t start_time, end_time;
    start_time = clock();
    int is_start = 0;
    while (d_limit < INT_MAX)
    {
        cout << "d_limit: " << d_limit << endl;
        while (!open.empty())
        {
            sub_state *current = open.top();
            open.pop();
            delete current;
        }
        if (is_start == 0)
        {
            is_start = 1;
        }
        else
        {
            start_state = new sub_state(start, target, 0);
        }
        int next_d_limit = INT_MAX;
        open.push(start_state);
        while (!open.empty())
        {
            sub_state *current = open.top();
            open.pop();
            if (current->get_d() > d_limit)
            {
                next_d_limit = min(next_d_limit, current->get_d());
            }
            else
            {
                if (current->is_over())
                {
                    const sub_state *temp = current;
                   // cout<<"g = "<<temp->get_g()<<endl;
                    while (temp->get_last_state() != NULL)
                    {
                        close += temp->get_last_move();
                        //cout<<temp->get_last_move()<<" "<<"h= "<<temp->get_h()<<"g= "<<temp->get_g()<<endl;
                        temp = temp->get_last_state();
                    }
                    for (int i = close.size() - 1; i >= 0; i--)
                    {
                        cout << close[i];
                    }
                    cout << endl;
                    end_time = clock();
                    cout << "time = " << double(end_time - start_time) / CLOCKS_PER_SEC << "s" << endl;
                    is_end = true;
                    break;
                }
                sub_state *up_state = current->move('U');
                sub_state *down_state = current->move('D');
                sub_state *left_state = current->move('L');
                sub_state *right_state = current->move('R');
                if (up_state->get_valid())
                {
                    open.push(up_state);
                }
                if (down_state->get_valid())
                {
                    open.push(down_state);
                }
                if (left_state->get_valid())
                {
                    open.push(left_state);
                }
                if (right_state->get_valid())
                {
                    open.push(right_state);
                }
            }
        }
        d_limit = next_d_limit;
        if (is_end)
            break;
    }
    //reverse(close.begin(), close.end());
   // outfile <<close<<","<<to_string(double(end_time - start_time) / CLOCKS_PER_SEC)<<endl;
}

void solution(void (*h)(const vector<vector<int>> &,
                        const vector<vector<int>> &),
              string input_start, string input_target)
{

    input_start = "../data/" + input_start;
    input_target = "../data/" + input_target;
    // cout<<"inputFile: "<<input_start<<endl;
    // cout<<"targetFile: "<<input_target<<endl;
    ifstream inputFile(input_start);
    ifstream targetFile(input_target);
    vector<vector<int>> start, target;
    string buffer;
    stringstream line;
    //处理文件
    while (getline(inputFile, buffer))
    {
        vector<int> w;
        string temp;
        w.clear();
        stringstream line(buffer);
        while (getline(line, temp, ' '))
        {
            w.push_back(stoi(temp));
        }
        start.push_back(w);
    }
    while (getline(targetFile, buffer))
    {
        vector<int> w;
        string temp;
        w.clear();
        stringstream line(buffer);
        while (getline(line, temp, ' '))
        {
            w.push_back(stoi(temp));
        }
        target.push_back(w);
    }

    h(start, target);
    inputFile.close();
    targetFile.close();
}

int main()
{
    string functionName;
    string inputFile;
    string targetFile;
    //依次输入三个参数
    cin >> functionName >> inputFile >> targetFile;
    if(functionName=="A_h1")
    {
        solution(A_h1,inputFile,targetFile);
    }
    else if(functionName=="IDA_h1")
    {
        solution(IDA_h1,inputFile,targetFile);
    }
    else if(functionName=="A_h2")
    {
        solution(A_h2,inputFile,targetFile);
    }
    else if(functionName=="IDA_h2")
    {
        solution(IDA_h2,inputFile,targetFile);

    }
    else
    {
        cout <<" functionName is wrong" << endl;
    }

    return 0;
}