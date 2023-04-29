import numpy as np


class TeacherSchedulingProblem:
    """This class encapsulates the Teacher Scheduling problem
    """

    def __init__(self, hard_constraint_penalty):
        """
        :param hard_constraint_penalty: the penalty factor for a hard-constraint violation
        """
        self.hard_constraint_penalty = hard_constraint_penalty

        # list of teachers:
        self.teachers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

        # teachers' respective shift preferences - morning, evening, night:
        self.shift_preference = [[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1]]

        # min and max number of teachers allowed for each shift - morning, evening, night:
        self.shift_min = [2, 2, 1]
        self.shift_max = [3, 4, 2]

        # max shifts per week allowed for each teacher
        self.max_shifts_per_week = 5

        # number of weeks we create a schedule for:
        self.weeks = 1

        # useful values:
        self.shift_per_day = len(self.shift_min)
        self.shifts_per_week = 7 * self.shift_per_day

    def __len__(self):
        """
        :return: the number of shifts in the schedule
        """
        return len(self.teachers) * self.shifts_per_week * self.weeks


    def get_cost(self, schedule):
        """
        Calculates the total cost of the various violations in the given schedule
        ...
        :param schedule: a list of binary values describing the given schedule
        :return: the calculated cost
        """

        if len(schedule) != self.__len__():
            raise ValueError("size of schedule list should be equal to ", self.__len__())

        # convert entire schedule into a dictionary with a separate schedule for each teacher:
        teacher_shifts_dict = self.get_teacher_shifts(schedule)

        # count the various violations:
        consecutive_shift_violations = self.count_consecutive_shift_violations(teacher_shifts_dict)
        shifts_per_week_violations = self.count_shifts_per_week_violations(teacher_shifts_dict)[1]
        teachers_per_shift_violations = self.count_teachers_per_shift_violations(teacher_shifts_dict)[1]
        shift_preference_violations = self.count_shift_preference_violations(teacher_shifts_dict)

        # calculate the cost of the violations:
        hard_contstraint_violations = consecutive_shift_violations + teachers_per_shift_violations + shifts_per_week_violations
        soft_contstraint_violations = shift_preference_violations

        return self.hard_constraint_penalty * hard_contstraint_violations + soft_contstraint_violations

    def get_teacher_shifts(self, schedule):
        """
        Converts the entire schedule into a dictionary with a separate schedule for each teacher
        :param schedule: a list of binary values describing the given schedule
        :return: a dictionary with each teacher as a key and the corresponding shifts as the value
        """
        shifts_per_teacher = self.__len__() // len(self.teachers)
        teacher_shifts_dict = {}
        shift_index = 0

        for teacher in self.teachers:
            teacher_shifts_dict[teacher] = schedule[shift_index:shift_index + shifts_per_teacher]
            shift_index += shifts_per_teacher

        return teacher_shifts_dict

    def count_consecutive_shift_violations(self, teacher_shifts_dict):
        """
        Counts the consecutive shift violations in the schedule
        :param teacher_shifts_dict: a dictionary with a separate schedule for each teacher
        :return: count of violations found
        """
        violations = 0
        # iterate over the shifts of each teacher:
        for teacher_shifts in teacher_shifts_dict.values():
            # look for two cosecutive '1's:
            for shift1, shift2 in zip(teacher_shifts, teacher_shifts[1:]):
                if shift1 == 1 and shift2 == 1:
                    violations += 1
        return violations

    def count_shifts_per_week_violations(self, teacher_shifts_dict):
        """
        Counts the max-shifts-per-week violations in the schedule
        :param teacher_shifts_dict: a dictionary with a separate schedule for each teacher
        :return: count of violations found
        """
        violations = 0
        weekly_shifts_list = []
        # iterate over the shifts of each teacher:
        for teacher_shifts in teacher_shifts_dict.values():  # all shifts of a single teacher
            # iterate over the shifts of each weeks:
            for i in range(0, self.weeks * self.shifts_per_week, self.shifts_per_week):
                # count all the '1's over the week:
                weekly_shifts = sum(teacher_shifts[i:i + self.shifts_per_week])
                weekly_shifts_list.append(weekly_shifts)
                if weekly_shifts > self.max_shifts_per_week:
                    violations += weekly_shifts - self.max_shifts_per_week

        return weekly_shifts_list, violations

    def count_teachers_per_shift_violations(self, teacher_shifts_dict):
        """
        Counts the number-of-teachers-per-shift violations in the schedule
        :param teacher_shifts_dict: a dictionary with a separate schedule for each teacher
        :return: count of violations found
        """
        # sum the shifts over all teachers:
        total_per_shift_list = [sum(shift) for shift in zip(*teacher_shifts_dict.values())]

        violations = 0
        # iterate over all shifts and count violations:
        for shift_index, num_of_teachers in enumerate(total_per_shift_list):
            daily_shift_index = shift_index % self.shift_per_day  # -> 0, 1, or 2 for the 3 shifts per day
            if (num_of_teachers > self.shift_max[daily_shift_index]):
                violations += num_of_teachers - self.shift_max[daily_shift_index]
            elif (num_of_teachers < self.shift_min[daily_shift_index]):
                violations += self.shift_min[daily_shift_index] - num_of_teachers

        return total_per_shift_list, violations

    def count_shift_preference_violations(self, teacher_shifts_dict):
        """
        Counts the teacher-preferences violations in the schedule
        :param teacher_shifts_dict: a dictionary with a separate schedule for each teacher
        :return: count of violations found
        """
        violations = 0
        for teacher_index, shift_preference in enumerate(self.shift_preference):
            # duplicate the shift-preference over the days of the period
            preference = shift_preference * (self.shifts_per_week // self.shift_per_day)
            # iterate over the shifts and compare to preferences:
            shifts = teacher_shifts_dict[self.teachers[teacher_index]]
            for pref, shift in zip(preference, shifts):
                if pref == 0 and shift == 1:
                    violations += 1

        return violations

    def print_schedule_info(self, schedule):
        """
        Prints the schedule and violations details
        :param schedule: a list of binary values describing the given schedule
        """
        teacher_shifts_dict = self.get_teacher_shifts(schedule)

        print("Schedule for each teacher:")
        for teacher in teacher_shifts_dict:  # all shifts of a single teacher
            print(teacher, ":", teacher_shifts_dict[teacher])

        print("consecutive shift violations = ", self.count_consecutive_shift_violations(teacher_shifts_dict))
        print()

        weekly_shifts_list, violations = self.count_shifts_per_week_violations(teacher_shifts_dict)
        print("Weekly Shifts = ", weekly_shifts_list)
        print("Shifts Per Week Violations = ", violations)
        print()

        total_per_shift_list, violations = self.count_teachers_per_shift_violations(teacher_shifts_dict)
        print("Teachers Per Shift = ", total_per_shift_list)
        print("Teachers Per Shift Violations = ", violations)
        print()

        shift_preference_violations = self.count_shift_preference_violations(teacher_shifts_dict)
        print("Shift Preference Violations = ", shift_preference_violations)
        print()


# testing the class:
def main():
    # create a problem instance:
    teachers = TeacherSchedulingProblem(10)

    random_solution = np.random.randint(2, size=len(teachers))
    print("Random Solution = ")
    print(random_solution)
    print()

    teachers.print_schedule_info(random_solution)

    print("Total Cost = ", teachers.get_cost(random_solution))


if __name__ == "__main__":
    main()
