import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, backref
from sqlalchemy import create_engine

Base = declarative_base()

class Experiment(Base):
    __tablename__ = 'experiments'
    experiment_id = Column(Integer, primary_key=True)
    dataset = Column(String(250), nullable=False)

class GridConfig(Base):
    __tablename__ = 'grid_configs'
    grid_config_id = Column(Integer, primary_key=True)
    type = Column(Integer, nullable=False)
    level = Column(Integer, nullable=False)
    T = Column(Integer, nullable=False)

class AdaptivityConfig(Base):
    __tablename__ = 'adaptivity_configs'
    adaptivity_config_id = Column(Integer, primary_key=True)

    num_refinements = Column(Integer, nullable=False)
    no_points = Column(Integer, nullable=True)
    percent = Column(Float, nullable=True)
    treshold = Column(Float, nullable=True)

class SolverConfig(Base):
    __tablename__ = 'solver_configs'
    solver_config_id = Column(Integer, primary_key=True)

    type = Column(Integer, nullable=False)
    max_iterations = Column(Integer, nullable=False)
    epsilon = Column(Float, nullable=False)
    treshold = Column(Float, nullable=False, server_default=0.0)

class RegularizationConfig(Base):
    __tablename__ = 'regularization_configs'
    regularization_config_id = Column(Integer, primary_key=True)

    lambda_reg = Column(Float, nullable=False)
    exponent_base = Column(Float, nullable=True)
    type = Column(Integer, nullable=False)

    def set_params(self, **params):
        if 'lambda_reg' in params:
            self.lambda_reg = params['lambda_reg']
        if 'exponent_base' in params:
            self.exponent_base = params['exponent_base']
        if 'type' in params:
            self.type = params['type']

    def get_params(self, deep=True):
        return {'lambda_reg': self.lambda_reg,
                'exponent_base': self.exponent_base,
                'type': self.type}

class Result(Base):
    __tablename__ = 'results'
    result_id = Column(Integer, primary_key=True, autoincrement=True)

    validation_mse = Column(Float, nullable=False)
    train_mse = Column(Float, nullable=True)
    test_mse = Column(Float, nullable=True)
    train_r2 = Column(Float, nullable=True)
    test_r2 = Column(Float, nullable=True)
    validation_grid_points_mean = Column(Float, nullable=True)
    validation_grid_points_stdev = Column(Float, nullable=True)
    train_grid_points = Column(Float, nullable=True)

    grid_config_id = Column(Integer, ForeignKey('grid_configs.grid_config_id'))
    grid_config = relationship(GridConfig, backref=backref('results', uselist=True))

    adaptivity_config_id = Column(Integer, ForeignKey('adaptivity_configs.adaptivity_config_id'))
    adaptivity_config = relationship(AdaptivityConfig, backref=backref('results', uselist=True))

    solver_config_id = Column(Integer, ForeignKey('solver_configs.solver_config_id'))
    solver_config = relationship(SolverConfig, foreign_keys=[solver_config_id])
    final_solver_config_id = Column(Integer, ForeignKey('solver_configs.solver_config_id'), nullable=True)
    final_solver_config = relationship(SolverConfig, foreign_keys=[final_solver_config_id])

    regularization_config_id = Column(Integer,
                                      ForeignKey('regularization_configs.regularization_config_id'))
    regularization_config = relationship(RegularizationConfig, backref=backref('results', uselist=True))

    experiment_id = Column(Integer, ForeignKey('experiments.experiment_id'))
    experiment = relationship(Experiment, backref=backref('results', uselist=True))

def make_session():
    package_dir= os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(package_dir, '../../experiments/results.db'))
    engine = create_engine('sqlite:///' + path)
    Base.metadata.create_all(engine)
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    return session
